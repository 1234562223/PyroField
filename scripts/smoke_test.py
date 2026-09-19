"""Quick sanity check that the forward model runs and produces sensible observations."""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyrofield.sim.synthetic import Scenario, forward, pack  # noqa: E402

TRUTH = {"R0": 0.10, "U": 4.0, "theta_w": 0.7854, "lb_k": 0.35, "w_buoy": 3.0, "Q": 1.0}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scen = Scenario(device=device)
    theta = pack(TRUTH, device=device)

    t0 = time.time()
    obs = forward(theta, scen)
    if device == "cuda":
        torch.cuda.synchronize()
    fwd_s = time.time() - t0

    print(f"device={device}  forward={fwd_s:.2f}s")
    for k, v in obs.items():
        print(
            f"  {k:6s} shape={tuple(v.shape)} "
            f"min={v.min().item():.4g} max={v.max().item():.4g} mean={v.mean().item():.4g}"
        )

    burned = (obs["mask"] > 0.5).float().mean(dim=(1, 2))
    print("  burned area fraction per mask snapshot:", [f"{b:.3f}" for b in burned.tolist()])

    # Gradient check: the whole chain must be differentiable w.r.t. the parameters.
    theta_g = theta.clone().requires_grad_(True)
    obs_g = forward(theta_g, scen)
    loss = obs_g["mask"].sum() + obs_g["plume"].sum() + obs_g["conc"].sum() * 1e3
    loss.backward()
    print("  d(loss)/d(theta) =", [f"{g:.4g}" for g in theta_g.grad.tolist()])
    if device == "cuda":
        print(f"  peak GPU mem: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")


if __name__ == "__main__":
    main()
