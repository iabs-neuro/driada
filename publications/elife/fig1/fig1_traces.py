"""
Fig 1 subfigure — Colorful neural activity traces.
Decorative panel for the eLife architecture figure.

Generates synthetic calcium-like traces (no data dependency).

Usage:
    python fig1_traces.py          # saves fig1_traces.pdf + .png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import elife_paths

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# ── Synthetic calcium trace generator ─────────────────────────────
rng = np.random.default_rng(42)

def synthetic_calcium(n_frames, fps=20, rate=0.08, tau_rise=0.1, tau_decay=1.5):
    """Generate a realistic calcium-like trace with random transients."""
    dt = 1.0 / fps
    t = np.arange(n_frames) * dt
    trace = np.zeros(n_frames)

    # Random spike times (Poisson-like)
    spike_prob = rate * dt
    spikes = rng.random(n_frames) < spike_prob

    # Convolve with calcium kernel
    kernel_len = int(5 * tau_decay * fps)
    k_t = np.arange(kernel_len) * dt
    kernel = (1 - np.exp(-k_t / tau_rise)) * np.exp(-k_t / tau_decay)
    kernel /= kernel.max()

    # Amplitudes vary per spike
    spike_signal = np.zeros(n_frames)
    spike_signal[spikes] = rng.uniform(0.3, 1.0, size=spikes.sum())

    trace = np.convolve(spike_signal, kernel, mode='full')[:n_frames]

    # Add small noise
    trace += rng.normal(0, 0.02, n_frames)
    trace = np.clip(trace, 0, None)

    return t, trace


# ── Parameters ────────────────────────────────────────────────────
n_neurons = 10
n_frames = 4000  # 200 seconds at 20 fps
fps = 20

# Colors — distinct palette for each neuron
COLORS = [
    "#E53935",  # red
    "#1E88E5",  # blue
    "#43A047",  # green
    "#FB8C00",  # orange
    "#8E24AA",  # purple
    "#00ACC1",  # cyan
    "#F4511E",  # deep orange
    "#3949AB",  # indigo
    "#7CB342",  # light green
    "#D81B60",  # pink
]

# ── Figure setup ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

spacing = 1.2  # vertical offset between traces

for i in range(n_neurons):
    t, trace = synthetic_calcium(
        n_frames, fps=fps,
        rate=rng.uniform(0.05, 0.15),
        tau_decay=rng.uniform(1.0, 2.0),
    )
    offset = (n_neurons - 1 - i) * spacing
    ax.plot(t, trace + offset, color=COLORS[i], linewidth=1.2, alpha=1.0)

    # Neuron label
    ax.text(-2, offset + 0.3, f"n{i+1}", fontsize=7, color=COLORS[i],
            fontweight="bold", va="center", ha="right")

ax.set_xlim(0, t[-1])
ax.set_ylim(-0.3, n_neurons * spacing + 0.5)
ax.set_xlabel("time, s", fontsize=9)
ax.set_ylabel("neural activity (ΔF/F)", fontsize=9)
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="x", labelsize=8)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────
fig.savefig(elife_paths.out_path("fig1_traces.pdf"), bbox_inches="tight", dpi=300)
fig.savefig(elife_paths.out_path("fig1_traces.png"), bbox_inches="tight", dpi=300)
print("Saved fig1_traces.pdf and .png")
