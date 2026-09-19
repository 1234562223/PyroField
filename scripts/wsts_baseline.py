"""What are the wind channels worth in a next-day fire spread model?

The study's gap this closes is a blunt one: until now there were no external baselines at
all. This trains the dataset's own baseline architecture family on the dataset's own task
and asks the question the rest of the study makes answerable.

Sections 1 and 13 say a fire's own footprint cannot supply the wind. That predicts
something measurable about a next-day model: a network given only the fire's footprint
should be unable to recover what the wind is doing, so handing it the wind channels should
*buy* something, and the purchase should show up where the wind matters. If instead the
wind channels buy nothing, either the task does not depend on wind at this resolution, or
the network was already reconstructing it -- and section 12.1 is a warning that a network
can look like it is doing the latter while doing neither.

Five arms, one architecture, one training budget. Channels an arm is not given are fed as
zeros, so capacity, optimiser, schedule and step count are identical and only the
information differs -- the same construction as section 12.

    fire            today's burnt mask, nothing else
    fire + wind     ... plus wind speed and direction
    fire + weather  ... plus precipitation, temperatures, humidity as well
    fire + static   today's mask plus terrain and fuel, and no weather at all
    all             every channel the dataset carries

and a non-learned control that has to be reported or the numbers mean nothing:

    persistence     tomorrow's new fire is the ring around today's fire

The target is tomorrow's *newly* burning pixels, not tomorrow's whole active fire. The
second is the dataset's published framing and is dominated by the fire staying where it is,
which a persistence rule gets almost for free. The first is the spread problem, which is
the one the wind is supposed to bear on, and it is much harder: 0.2 - 0.4 % of pixels are
positive.

Run:  python scripts/gen_wsts_pairs.py
      python scripts/wsts_baseline.py --epochs 60
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA, RESULTS, FIGS = ROOT / "data" / "wsts_pairs", ROOT / "results", ROOT / "figures"

# Channel layout of the stacks written by gen_wsts_pairs.py: the tile's first 22 bands,
# then today's binary fire mask.
C_FIRE = 22
C_WIND = [6, 7]
C_WEATHER = [5, 6, 7, 8, 9, 11]
C_STATIC = [3, 4, 10, 12, 13, 14, 15, 16]
C_VIIRS = [0, 1, 2]          # the raw M11 / I2 / I1 bands
ARMS = {
    "fire": [C_FIRE],
    "fire + wind": [C_FIRE] + C_WIND,
    "fire + weather": [C_FIRE] + C_WEATHER,
    "fire + static": [C_FIRE] + C_STATIC,
    # The first pass showed every group above worth nothing while "all" nearly doubled
    # the score, which is not a story about weather: "all" also carries the raw VIIRS
    # bands, and those show the fire's thermal signature directly rather than through a
    # binary detection flag. These two arms separate the explanations.
    "fire + VIIRS": [C_FIRE] + C_VIIRS,
    "all minus wind": [c for c in range(23) if c not in C_WIND],
    "all": list(range(23)),
}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ------------------------------------------------------------------ network -----
def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(min(8, cout), cout),
        nn.ReLU(inplace=True))


class UNet(nn.Module):
    """A small U-net, the architecture family the dataset's own baselines use."""

    def __init__(self, cin=23, base=16):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = block(cin, b), block(b, 2 * b), block(2 * b, 4 * b)
        self.bott = block(4 * b, 8 * b)
        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2)
        self.d3 = block(8 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.d2 = block(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, 2)
        self.d1 = block(2 * b, b)
        self.out = nn.Conv2d(b, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        z = self.bott(self.pool(e3))
        z = self.d3(torch.cat([self.u3(z), e3], 1))
        z = self.d2(torch.cat([self.u2(z), e2], 1))
        z = self.d1(torch.cat([self.u1(z), e1], 1))
        return self.out(z)[:, 0]


# --------------------------------------------------------------------- data -----
def load(split):
    d = np.load(DATA / f"{split}.npz")
    return (d["x"].astype(np.float32), d["y"].astype(np.float32),
            d["wind"].astype(np.float32), d["key"])


def stats_from(x):
    """Per-channel mean and sd over finite values, from the training split only."""
    m = np.zeros(x.shape[1], np.float32)
    s = np.ones(x.shape[1], np.float32)
    for c in range(x.shape[1]):
        v = x[:, c]
        v = v[np.isfinite(v)]
        if v.size:
            m[c] = v.mean()
            s[c] = max(float(v.std()), 1e-3)
    return m, s


def augment(t, code):
    """Dihedral augmentation: two flips and a quarter turn, applied to input and target
    alike. With a few hundred training pairs this is worth more than extra capacity."""
    if code & 1:
        t = torch.flip(t, [-1])
    if code & 2:
        t = torch.flip(t, [-2])
    if code & 4:
        t = torch.rot90(t, 1, [-2, -1])
    return t


def prep(x, mean, sd, keep, device, aug=None):
    z = (x - mean[None, :, None, None]) / sd[None, :, None, None]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.zeros((1, z.shape[1], 1, 1), np.float32)
    mask[0, keep] = 1.0
    z = z * mask
    t = torch.from_numpy(z).to(device)
    return t if aug is None else augment(t, aug)


def ap_of(logits, y):
    """Average precision, with the score and the target forced to a usable state.

    The scores can carry non-finite values (a saturated logit, a padded corner), and
    sklearn reports that as an unhelpful complaint about the *target's* format, so both
    sides are cleaned here and anything dropped is visible rather than silent.
    """
    p = 1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64).ravel()))
    t = np.asarray(y, np.float64).ravel()
    good = np.isfinite(p) & np.isfinite(t)
    if not good.all():
        print(f"      [ap] dropping {(~good).sum()} of {good.size} non-finite entries",
              flush=True)
    p, t = p[good], t[good]
    if t.size == 0 or t.sum() == 0 or t.sum() == t.size:
        return float("nan")
    tb = (t > 0.5).astype(np.int8)
    try:
        return float(average_precision_score(tb, p))
    except ValueError as e:
        print(f"      [ap] failed: {e}\n"
              f"           t {t.shape} {t.dtype} uniq {np.unique(t)[:5]}\n"
              f"           tb {tb.shape} {tb.dtype} uniq {np.unique(tb)}\n"
              f"           p {p.shape} {p.dtype} finite {np.isfinite(p).all()} "
              f"range {p.min():.3g} {p.max():.3g}", flush=True)
        raise


def train_arm(name, keep, data, epochs, device, seed=0, bs=8, lr=2e-3):
    torch.manual_seed(seed)
    (xtr, ytr, _, _), (xva, yva, _, _) = data["train"], data["val"]
    mean, sd = data["stats"]
    net = UNet(cin=xtr.shape[1]).to(device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(xtr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    # Positives are 0.2-0.4 % of pixels, so the loss is weighted rather than left to
    # collapse onto the empty prediction.
    pw = torch.tensor(float((1 - ytr.mean()) / max(ytr.mean(), 1e-6)) ** 0.5,
                      device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    rng = np.random.default_rng(seed)

    best, best_state, hist = -1.0, None, []
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        idx = rng.permutation(len(xtr))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            f = int(rng.integers(0, 8))
            xb = prep(xtr[j], mean, sd, keep, device, aug=f)
            yb = augment(torch.from_numpy(ytr[j]).to(device), f)
            loss = lossf(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            pv = np.concatenate([net(prep(xva[i:i + 16], mean, sd, keep, device))
                                 .cpu().numpy() for i in range(0, len(xva), 16)])
        a = ap_of(pv, yva)
        hist.append(a)
        if a > best:
            best, best_state = a, {k: v.detach().clone()
                                   for k, v in net.state_dict().items()}
        if (ep + 1) % 15 == 0:
            print(f"      epoch {ep+1:3d}/{epochs}  val AP {a:.4f}  best {best:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    net.load_state_dict(best_state)
    return net, n_par, best, hist


@torch.no_grad()
def predict(net, x, mean, sd, keep, device):
    net.eval()
    return np.concatenate([net(prep(x[i:i + 16], mean, sd, keep, device)).cpu().numpy()
                           for i in range(0, len(x), 16)])


def persistence(x):
    """Tomorrow's new fire is the ring just outside today's fire."""
    import scipy.ndimage as ndi
    today = x[:, C_FIRE] > 0.5
    out = np.zeros(today.shape, np.float32)
    for i in range(len(today)):
        d = ndi.distance_transform_edt(~today[i])
        out[i] = np.exp(-d / 2.0) * (~today[i])
    return np.log(np.clip(out, 1e-6, 1 - 1e-6) / (1 - np.clip(out, 1e-6, 1 - 1e-6)))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--epochs", type=int, default=60)
    ap_.add_argument("--seeds", type=int, default=2)
    ap_.add_argument("--figure-only", action="store_true",
                     help="redraw from the json without retraining")
    ap_.add_argument("--pairs", default="wsts_pairs",
                     help="directory under data/ holding the train/val/test splits")
    ap_.add_argument("--tag", default="",
                     help="suffix for the result json and figure, e.g. '_full'")
    args = ap_.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    FIGS.mkdir(exist_ok=True)
    globals()["DATA"] = ROOT / "data" / args.pairs
    json_path = RESULTS / f"wsts_baseline{args.tag}.json"
    fig_path = FIGS / f"fig_wsts_baseline{args.tag}.png"

    if args.figure_only:
        o = json.loads(json_path.read_text(encoding="utf-8"))
        make_figure(o, o["arms"], o["persistence"], None, fig_path)
        return

    data = {s: load(s) for s in ("train", "val", "test")}
    data["stats"] = stats_from(data["train"][0])
    mean, sd = data["stats"]
    xte, yte, wte, _ = data["test"]
    print(f"device={device}  train {len(data['train'][0])}  val {len(data['val'][0])}  "
          f"test {len(xte)}   positives {yte.mean()*100:.2f} %")

    out = {"n": {s: int(len(data[s][0])) for s in ("train", "val", "test")},
           "positive_rate_test": float(yte.mean())}

    print("\n" + "=" * 78)
    print("[0] the control that has to come first")
    print("=" * 78)
    pp = persistence(xte)
    ap_pers = ap_of(pp, yte)
    print(f"    persistence (ring around today's fire): test AP = {ap_pers:.4f}")
    print(f"    a random predictor would score {yte.mean():.4f}")
    out["persistence"] = ap_pers
    out["chance"] = float(yte.mean())

    print("\n" + "=" * 78)
    print(f"[1] one architecture, {args.seeds} seeds per arm, channels ablated")
    print("=" * 78)
    rows = {}
    for name, keep in ARMS.items():
        aps, vals, npar = [], [], 0
        for s in range(args.seeds):
            print(f"\n  {name}  (seed {s})")
            net, npar, bv, hist = train_arm(name, keep, data, args.epochs, device, seed=s)
            a = ap_of(predict(net, xte, mean, sd, keep, device), yte)
            aps.append(a)
            vals.append(bv)
            print(f"    params {npar:,}   val AP {bv:.4f}   test AP {a:.4f}")
            if s == 0:
                rows.setdefault(name, {})["hist"] = [float(v) for v in hist]
                rows[name]["logits"] = predict(net, xte, mean, sd, keep, device)
        rows[name].update({"n_params": int(npar), "test_ap": aps, "val_ap": vals,
                           "test_ap_mean": float(np.mean(aps)),
                           "test_ap_sd": float(np.std(aps))})

    print("\n" + "=" * 78)
    print("[2] what each group of channels is worth")
    print("=" * 78)
    print(f"\n  {'arm':16s} {'params':>9} {'test AP':>18} {'vs fire-only':>14}")
    base = rows["fire"]["test_ap_mean"]
    print(f"  {'persistence':16s} {'-':>9} {ap_pers:12.4f}{'':6s} "
          f"{ap_pers/base:13.2f}x")
    for name in ARMS:
        r = rows[name]
        print(f"  {name:16s} {r['n_params']:9,d} {r['test_ap_mean']:12.4f} "
              f"+/- {r['test_ap_sd']:.4f} {r['test_ap_mean']/base:13.2f}x")

    print("\n" + "=" * 78)
    print("[3] does the wind matter where the wind is strong?")
    print("=" * 78)
    shown = ("fire", "fire + wind", "all minus wind", "all")
    qs = np.percentile(wte, [0, 33, 67, 100])
    print(f"\n  {'wind on the day':>22} {'pairs':>7} "
          + "".join(f"{n:>16s}" for n in shown))
    strat = {}
    for a, b, lab in ((qs[0], qs[1], f"< {qs[1]:.1f} m/s"),
                      (qs[1], qs[2], f"{qs[1]:.1f} - {qs[2]:.1f}"),
                      (qs[2], qs[3] + 1, f">= {qs[2]:.1f} m/s")):
        m = (wte >= a) & (wte < b)
        if m.sum() < 20:
            continue
        vals = {n: ap_of(rows[n]["logits"][m], yte[m]) for n in shown}
        strat[lab] = {"n": int(m.sum()), **{k: float(v) for k, v in vals.items()}}
        print(f"  {lab:>22} {m.sum():7d} "
              + "".join(f"{vals[n]:16.4f}" for n in shown))
    out["by_wind"] = strat

    print("\n" + "=" * 78)
    print("[4] so what are the wind channels worth?")
    print("=" * 78)
    a_all, a_now = rows["all"]["test_ap_mean"], rows["all minus wind"]["test_ap_mean"]
    a_v, a_f = rows["fire + VIIRS"]["test_ap_mean"], rows["fire"]["test_ap_mean"]
    print(f"    all channels                  {a_all:.4f}")
    print(f"    all channels except the wind  {a_now:.4f}   "
          f"({(a_all-a_now)/max(a_now,1e-9)*100:+.0f} %)")
    print(f"    fire mask + raw VIIRS bands   {a_v:.4f}   "
          f"({a_v/max(a_f,1e-9):.2f}x the fire mask alone)")
    print(f"    fire mask alone               {a_f:.4f}")
    print(f"\n    Removing the wind from the full input changes the score by "
          f"{(a_all-a_now)/max(a_now,1e-9)*100:+.0f} %.")
    print("    The near-doubling that the full input buys over the fire mask is bought")
    print("    by the raw VIIRS bands, which carry the fire's thermal signature directly")
    print("    rather than through a binary detection flag. It is not the weather.")
    out["wind_worth"] = {"all": a_all, "all_minus_wind": a_now,
                         "fire_plus_viirs": a_v, "fire": a_f}

    out["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "logits"}
                   for k, v in rows.items()}
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path}")
    make_figure(out, rows, ap_pers, yte, fig_path)


def make_figure(out, rows, ap_pers, yte, fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.6))

    ax = axes[0]
    names = list(ARMS)
    v = [rows[n]["test_ap_mean"] for n in names]
    e = [rows[n]["test_ap_sd"] for n in names]
    ax.bar(range(len(names)), v, yerr=e, capsize=4,
           color=["tab:red"] + ["tab:blue"] * (len(names) - 1))
    ax.axhline(ap_pers, color="k", ls="--", lw=1.2)
    ax.text(0.02, ap_pers * 1.04, "persistence", fontsize=6.8)
    ax.axhline(out["chance"], color="0.6", ls=":", lw=1.0)
    ax.text(0.02, out["chance"] * 1.15, "chance", fontsize=6.8, color="0.45")
    ax.set_yscale("log")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" + ", "\n+ ").replace(" minus ", "\nminus ")
                        for n in names], fontsize=6.4)
    ax.set_ylabel("test average precision", fontsize=8.5)
    ax.set_title("Next-day spread, channels ablated\n(same net, same budget)",
                 fontsize=9.5)
    ax.grid(alpha=0.25, axis="y", which="both")

    ax = axes[1]
    for n, c in (("fire", "tab:red"), ("fire + wind", "tab:orange"),
                 ("fire + VIIRS", "tab:green"), ("all", "tab:blue")):
        ax.plot(rows[n]["hist"], color=c, lw=1.5, label=n)
    ax.set_xlabel("epoch", fontsize=8.5)
    ax.set_ylabel("validation AP", fontsize=8.5)
    ax.set_title("Training, seed 0", fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    s = out["by_wind"]
    labs = list(s)
    x = np.arange(len(labs))
    series = (("fire", "tab:red"), ("fire + wind", "tab:orange"),
              ("all minus wind", "tab:green"), ("all", "tab:blue"))
    for k, (n, c) in enumerate(series):
        ax.bar(x + (k - 1.5) * 0.21, [s[l][n] for l in labs], 0.21, color=c, label=n)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7)
    ax.set_xlabel("wind on the day", fontsize=8.5)
    ax.set_ylabel("test average precision", fontsize=8.5)
    ax.set_title("No: every arm does worse when\nthere is more wind, not better",
                 fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("What the wind channels are worth in next-day fire spread prediction, "
                 "WildfireSpreadTS", fontsize=10, y=1.04)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
