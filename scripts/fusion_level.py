"""Where you fuse: feature level versus physical-state level.

This is the comparison the project plan called its make-or-break experiment, and until now
it had not been run. The claim under test is that fusing sensors at the level of a shared
*physical state* -- inverting them through the fire model -- behaves differently from
fusing them at the level of learned features, and that the difference is not accuracy on
held-out data but what happens when the data moves.

Four arms, two axes:

                        mask only            mask + plume
  feature level    CNN, plume input zeroed   CNN, both inputs
  state level      LM inversion              LM inversion

The two CNN arms are the *same network with the same capacity and the same training
budget*; the mask-only arm simply receives zeros on the plume branch. Nothing differs but
the information available, which is what makes it an ablation rather than a comparison of
two models.

The test set is split in wind speed. In-distribution matches training (2-5 m/s);
out-of-distribution does not (5.5-8 m/s). That split is the whole experiment. A network
can score well in distribution on a quantity the data does not contain, by learning the
distribution the training fires were drawn from. Only moving the test distribution
separates having learned the physics from having learned the sampling.

Run:  python scripts/gen_dataset.py
      python scripts/fusion_level.py --epochs 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data", ROOT / "results", ROOT / "figures"

from pyrofield.eval.inversion import levenberg_marquardt  # noqa: E402
from pyrofield.sim.synthetic import (  # noqa: E402
    LOG_PARAMS,
    PARAM_NAMES,
    Scenario,
    add_noise,
    forward,
    pack,
    unpack,
)

MASK_TIMES = (17, 35, 53, 71, 89, 109)
PLUME_TIMES = (53, 109)
IU, ITW = PARAM_NAMES.index("U"), PARAM_NAMES.index("theta_w")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ----------------------------------------------------------------- network -----
def norm(c):
    """GroupNorm, not BatchNorm.

    The mask-only arm feeds an all-zero tensor to the plume branch, and a BatchNorm
    whose running variance collapses to zero behaves differently in train and eval
    mode. A first attempt with BatchNorm diverged during the high-learning-rate phase
    (validation loss 0.76 -> 1164 -> 3.2). GroupNorm has no running statistics, so
    train and eval agree by construction.
    """
    return nn.GroupNorm(min(8, c), c)


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), norm(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), norm(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class FeatureFusion(nn.Module):
    """Two conv encoders, concatenated features, one regression head.

    The textbook feature-level fusion architecture, which is the thing being compared
    against. Wind direction is predicted as (cos, sin) rather than as an angle, so the
    network is not penalised for the branch cut.
    """

    def __init__(self):
        super().__init__()
        self.mask_enc = nn.Sequential(block(len(MASK_TIMES), 32), block(32, 64),
                                      block(64, 128), block(128, 128),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.plume_enc = nn.Sequential(block(len(PLUME_TIMES), 32), block(32, 64),
                                       block(64, 128),
                                       nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(256, 256), nn.ReLU(inplace=True),
                                  nn.Linear(256, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, len(PARAM_NAMES) + 1))

    def forward(self, mask, plume):
        return self.head(torch.cat([self.mask_enc(mask), self.plume_enc(plume)], dim=1))


def to_targets(params: np.ndarray) -> np.ndarray:
    """Working-space parameters, with the wind direction split into (cos, sin)."""
    out = np.zeros((len(params), len(PARAM_NAMES) + 1), np.float32)
    j = 0
    for i, name in enumerate(PARAM_NAMES):
        if name == "theta_w":
            out[:, j] = np.cos(params[:, i])
            out[:, j + 1] = np.sin(params[:, i])
            j += 2
        else:
            out[:, j] = params[:, i]
            j += 1
    return out


def from_outputs(y: np.ndarray) -> np.ndarray:
    """Inverse of :func:`to_targets`."""
    out = np.zeros((len(y), len(PARAM_NAMES)), np.float32)
    j = 0
    for i, name in enumerate(PARAM_NAMES):
        if name == "theta_w":
            out[:, i] = np.arctan2(y[:, j + 1], y[:, j])
            j += 2
        else:
            out[:, i] = y[:, j]
            j += 1
    return out


def load(split):
    d = np.load(DATA / f"fusion_{split}.npz")
    return d["mask"], d["plume"], d["params"]


def batches(n, bs, rng=None):
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def prep(mask, plume, idx, device, use_plume):
    m = torch.from_numpy(mask[idx]).float().div_(255).to(device)
    p = torch.from_numpy(plume[idx]).float().div_(255).to(device)
    if not use_plume:
        p = torch.zeros_like(p)
    return m, p


def train_arm(use_plume, epochs, device, seed=0, bs=32, lr=6e-4):
    torch.manual_seed(seed)
    tr_m, tr_p, tr_y = load("train")
    va_m, va_p, va_y = load("val")
    ty = torch.from_numpy(to_targets(tr_y)).to(device)
    vy = torch.from_numpy(to_targets(va_y)).to(device)
    # Normalise targets so no component dominates the loss purely by its units.
    mu, sd = ty.mean(0, keepdim=True), ty.std(0, keepdim=True).clamp(min=1e-6)

    net = FeatureFusion().to(device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * math.ceil(len(tr_m) / bs))
    rng = np.random.default_rng(seed)

    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        for idx in batches(len(tr_m), bs, rng):
            m, p = prep(tr_m, tr_p, idx, device, use_plume)
            loss = nn.functional.mse_loss(net(m, p), (ty[idx] - mu) / sd)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for idx in batches(len(va_m), 64):
                m, p = prep(va_m, va_p, idx, device, use_plume)
                vl += nn.functional.mse_loss(net(m, p), (vy[idx] - mu) / sd,
                                             reduction="sum").item()
            vl /= len(va_m) * vy.shape[1]
        hist.append(vl)
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"      epoch {ep+1:3d}/{epochs}  val {vl:.5f}  best {best:.5f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    return net, mu, sd, n_par, hist


@torch.no_grad()
def predict(net, mu, sd, split, device, use_plume):
    m_, p_, y_ = load(split)
    net.eval()
    preds = []
    for idx in batches(len(m_), 64):
        m, p = prep(m_, p_, idx, device, use_plume)
        preds.append((net(m, p) * sd + mu).cpu().numpy())
    return from_outputs(np.concatenate(preds)), y_


def errors(pred, true):
    """Relative error for log parameters, degrees for the wind direction."""
    out = {}
    for i, (name, is_log) in enumerate(zip(PARAM_NAMES, LOG_PARAMS)):
        d = pred[:, i] - true[:, i]
        if is_log:
            out[name] = np.abs(np.expm1(d))
        else:
            ang = (d + np.pi) % (2 * np.pi) - np.pi
            out[name] = np.abs(np.degrees(ang))
    return out


def state_level(split, n_cases, device, use_plume, seed=0):
    """The state-level arm: invert the observations through the physics."""
    keys = ("mask", "plume") if use_plume else ("mask",)
    sc = Scenario(device=device, n_steps=110, mask_times=MASK_TIMES,
                  plume_times=PLUME_TIMES, conc_times=())
    _, _, y_ = load(split)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(y_), size=min(n_cases, len(y_)), replace=False)
    gen = torch.Generator(device=device).manual_seed(seed + 7)

    pred, true, t0 = [], [], time.time()
    for k, i in enumerate(pick):
        th = torch.from_numpy(y_[i]).to(device)
        target = add_noise(forward(th, sc), gen)
        start = th.clone()
        for j in range(len(PARAM_NAMES)):
            start[j] = start[j] + float(rng.uniform(-0.35, 0.35))
        res = levenberg_marquardt(start, th, target, sc, keys,
                                  schedule=((4.0, 5), (2.0, 4), (0.0, 10)))
        pred.append(res.theta_hat.cpu().numpy())
        true.append(y_[i])
        if (k + 1) % 5 == 0:
            el = time.time() - t0
            print(f"      {split} {'M+P' if use_plume else 'M':4s} "
                  f"{k+1}/{len(pick)}  ({el:.0f}s)", flush=True)
    return np.array(pred), np.array(true)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lm-cases", type=int, default=25)
    ap.add_argument("--figure-only", action="store_true",
                    help="rebuild the figure from results/fusion_preds.npz")
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    # Every arm's per-case (predicted, true) parameters, kept so the figure can be
    # rebuilt without repeating 74 inversions.
    preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    if args.figure_only:
        z = np.load(RESULTS / "fusion_preds.npz")
        for k in z.files:
            if k.endswith("__pred"):
                tag, split = k[:-6].rsplit("|", 1)
                preds[(tag, split)] = (z[k], z[f"{tag}|{split}__true"])
        out = json.loads((RESULTS / "fusion_level.json").read_text(encoding="utf-8"))
        make_figure(preds, out["summary"], out)
        return

    print("=" * 78)
    print("[1] feature-level fusion: one network, trained twice")
    print("=" * 78)
    nets = {}
    for use_plume, tag in ((False, "CNN mask only"), (True, "CNN mask + plume")):
        print(f"\n  {tag}")
        net, mu, sd, n_par, hist = train_arm(use_plume, args.epochs, device)
        nets[tag] = (net, mu, sd, use_plume)
        print(f"    parameters: {n_par:,}   best val {min(hist):.5f}")
        out[tag] = {"n_params": int(n_par), "val_best": float(min(hist)),
                    "val_hist": [float(v) for v in hist]}

    print()
    print("=" * 78)
    print("[2] what each arm gets, in and out of distribution")
    print("=" * 78)
    rows = {}
    for tag, (net, mu, sd, use_plume) in nets.items():
        for split in ("test_id", "test_ood"):
            pred, true = predict(net, mu, sd, split, device, use_plume)
            preds[(tag, split)] = (pred, true)
            e = errors(pred, true)
            rows[(tag, split)] = {k: float(np.median(v)) for k, v in e.items()}
            # Does it just predict the training mean?
            bias_U = float(np.median(np.expm1(pred[:, IU] - true[:, IU])))
            rows[(tag, split)]["U_bias"] = bias_U
            rows[(tag, split)]["U_pred_median"] = float(np.median(np.exp(pred[:, IU])))
            rows[(tag, split)]["U_true_median"] = float(np.median(np.exp(true[:, IU])))

    # The null model: ignore the images entirely and return the training median of every
    # parameter. Anything a network gets on top of this came from the data; anything it
    # loses relative to this out of distribution came from the training prior.
    _, _, tr_y = load("train")
    prior = np.median(tr_y, axis=0)
    for split in ("test_id", "test_ood"):
        _, _, y_ = load(split)
        e = errors(np.repeat(prior[None], len(y_), 0), y_)
        rows[("prior (train median)", split)] = {k: float(np.median(v))
                                                 for k, v in e.items()}
        rows[("prior (train median)", split)]["U_pred_median"] = float(np.exp(prior[IU]))
        rows[("prior (train median)", split)]["U_true_median"] = float(
            np.median(np.exp(y_[:, IU])))

    print(f"\n  {'arm':22s} {'split':9s} {'U err':>8s} {'R0 err':>8s} "
          f"{'theta_w':>9s} {'U pred':>8s} {'U true':>8s}")
    for tag in ["prior (train median)", *nets]:
        for split in ("test_id", "test_ood"):
            r = rows[(tag, split)]
            print(f"  {tag:22s} {split:9s} {r['U']*100:7.1f}% {r['R0']*100:7.1f}% "
                  f"{r['theta_w']:8.2f}d {r['U_pred_median']:8.2f} "
                  f"{r['U_true_median']:8.2f}")

    print()
    print("=" * 78)
    print(f"[3] state-level fusion: inversion through the physics ({args.lm_cases} cases)")
    print("=" * 78)
    for use_plume, tag in ((True, "state mask + plume"), (False, "state mask only")):
        n = args.lm_cases if use_plume else max(args.lm_cases // 2, 8)
        for split in ("test_id", "test_ood"):
            print(f"\n  {tag}, {split}")
            pred, true = state_level(split, n, device, use_plume)
            preds[(tag, split)] = (pred, true)
            e = errors(pred, true)
            rows[(tag, split)] = {k: float(np.median(v)) for k, v in e.items()}
            rows[(tag, split)]["U_pred_median"] = float(np.median(np.exp(pred[:, IU])))
            rows[(tag, split)]["U_true_median"] = float(np.median(np.exp(true[:, IU])))
            r = rows[(tag, split)]
            print(f"    U err {r['U']*100:.1f}%   R0 err {r['R0']*100:.1f}%   "
                  f"theta_w {r['theta_w']:.2f} deg")

    print()
    print("=" * 78)
    print("[4] the 2x2, on wind speed")
    print("=" * 78)
    print(f"  {'':22s} {'in-distribution':>18s} {'out-of-distribution':>22s} "
          f"{'degradation':>13s}")
    summary = {}
    for tag in ("prior (train median)", "CNN mask only", "CNN mask + plume",
                "state mask only", "state mask + plume"):
        a = rows.get((tag, "test_id"), {}).get("U")
        b = rows.get((tag, "test_ood"), {}).get("U")
        if a is None or b is None:
            continue
        summary[tag] = {"id": a, "ood": b, "ratio": b / max(a, 1e-9)}
        print(f"  {tag:22s} {a*100:17.1f}% {b*100:21.1f}% {b/max(a,1e-9):12.1f}x")

    out["rows"] = {f"{k[0]}|{k[1]}": v for k, v in rows.items()}
    out["summary"] = summary
    (RESULTS / "fusion_level.json").write_text(json.dumps(out, indent=2),
                                               encoding="utf-8")
    print(f"\nwrote {RESULTS/'fusion_level.json'}")

    np.savez_compressed(RESULTS / "fusion_preds.npz",
                        **{f"{t}|{s}__{w}": a
                           for (t, s), pt in preds.items()
                           for w, a in (("pred", pt[0]), ("true", pt[1]))})
    print(f"wrote {RESULTS/'fusion_preds.npz'}")

    make_figure(preds, summary, out)


def make_figure(preds, summary, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5))

    ax = axes[0]
    tags = [t for t in ("prior (train median)", "CNN mask only", "CNN mask + plume",
                        "state mask only", "state mask + plume") if t in summary]
    x = np.arange(len(tags))
    ax.bar(x - 0.2, [summary[t]["id"] * 100 for t in tags], 0.4,
           color="tab:blue", label="in-distribution")
    ax.bar(x + 0.2, [summary[t]["ood"] * 100 for t in tags], 0.4,
           color="tab:red", label="out-of-distribution")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" mask", "\nmask").replace(" (train median)",
                                                             "\n(train median)")
                        for t in tags], fontsize=6.8)
    ax.set_ylabel("median error in wind speed (%)", fontsize=8.5)
    ax.set_title("Same network, same budget;\nonly the information differs", fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(alpha=0.25, axis="y", which="both")

    ax = axes[1]
    lims = [1.5, 8.5]
    ax.axvspan(5.0, 8.5, color="0.9", zorder=0)
    for tag, c, lab in (("CNN mask only", "tab:red", "CNN, mask only"),
                        ("CNN mask + plume", "tab:blue", "CNN, mask + plume")):
        for split, mk in (("test_id", "o"), ("test_ood", "^")):
            if (tag, split) not in preds:
                continue
            pred, true = preds[(tag, split)]
            ax.scatter(np.exp(true[:, IU]), np.exp(pred[:, IU]), s=6, alpha=0.30,
                       marker=mk, color=c, zorder=2,
                       label=lab if split == "test_id" else None)
    for split, mk in (("test_id", "o"), ("test_ood", "^")):
        key = ("state mask + plume", split)
        if key not in preds:
            continue
        pred, true = preds[key]
        ax.scatter(np.exp(true[:, IU]), np.exp(pred[:, IU]), s=34, marker=mk,
                   facecolor="none", edgecolor="k", linewidth=1.2, zorder=4,
                   label="state level, mask + plume" if split == "test_id" else None)
    ax.plot(lims, lims, "k--", lw=1.1, zorder=3)
    ax.text(5.15, 2.0, "outside\ntraining", fontsize=7, color="0.35")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("true wind (m/s)", fontsize=8.5)
    ax.set_ylabel("estimated wind (m/s)", fontsize=8.5)
    ax.set_title("Outside the training range the networks\nstop tracking; the inversion does not",
                 fontsize=9.5)
    ax.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax = axes[2]
    for tag, c in (("CNN mask only", "tab:red"), ("CNN mask + plume", "tab:blue")):
        if tag in out:
            ax.plot(out[tag]["val_hist"], color=c, lw=1.6, label=tag)
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=8.5)
    ax.set_ylabel("validation loss (normalised)", fontsize=8.5)
    ax.set_title("Both train to convergence", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("Fusing at the feature level versus at the level of the physical state",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    p = FIGS / "fig_fusion_level.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
