"""
Does subtracting the environmental-noise mic help?

dev_1 sits where it mostly picks up environmental / handling noise (little heart
sound); dev_2 picks up heart sound + the same noise.  So x_dev2 - x_dev1 is a
crude analog noise-canceller applied *before* the MM denoiser.  This script
compares the two front-ends on identical sessions:

  A.  single  : y = x_dev2                 (demo_full.py        -> *.wav)
  B.  diff    : y = x_dev2 - x_dev1         (demo_full_diff.py   -> *_diff.wav)

Both go through the same band-pass -> reference-align -> MM denoise -> overlap-add
pipeline.  We then score the *denoised* output of each with the reference-free
PCG quality metrics from analyze.py and report which front-end wins, per metric.

Run demo_full.py / demo_full_diff.py first for the sessions you want compared.

Usage:
    python compare_diff.py
    python compare_diff.py --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import welch

from analyze import (
    FS, PCG_LO, PCG_HI, HR_LAG_LO_S, HR_LAG_HI_S,
    load_wav, compute_metrics, shannon_envelope, envelope_autocorr,
)


# metric key -> (display name, "is higher better?")
# (denoised-signal quality metrics; NRR is just energy removed by MM, not a
#  quality measure, so it is shown for context but not scored.)
QUALITY_METRICS = [
    ("SNR_env denoised (dB)",     "SNR_env  [dB]",      True),
    ("SNR_pk denoised (dB)",      "SNR_pk   [dB]",      True),
    ("SNR_band denoised (dB)",    "SNR_band [dB]",      True),
    ("periodicity denoised",      "periodicity SQI",    True),
    ("kurtosis denoised",         "kurtosis (S1/S2)",   True),
    ("spectral conc denoised",    "spectral conc",      True),
]


# ---------------------------------------------------------------------------
def session_paths(base_dir: str) -> dict:
    return {
        "single": (os.path.join(base_dir, "noisy.wav"),
                   os.path.join(base_dir, "denoised_mm.wav")),
        "diff":   (os.path.join(base_dir, "noisy_diff.wav"),
                   os.path.join(base_dir, "denoised_mm_diff.wav")),
    }


def fmt_val(key: str, v: float) -> str:
    if "conc" in key:
        return f"{v:.1%}"
    if "periodicity" in key:
        return f"{v:.3f}"
    if "kurtosis" in key:
        return f"{v:.1f}"
    return f"{v:+.2f}"


def compare_session(label: str, base_dir: str) -> dict | None:
    paths = session_paths(base_dir)
    for tag, (n, d) in paths.items():
        if not (os.path.exists(n) and os.path.exists(d)):
            print(f"[compare] skipping {label}: missing {tag} wavs "
                  f"({n} / {d}) -- run the corresponding demo first.")
            return None

    out = {}
    for tag, (n_path, d_path) in paths.items():
        noisy    = load_wav(n_path)
        denoised = load_wav(d_path)
        out[tag] = {"noisy": noisy, "denoised": denoised,
                    "metrics": compute_metrics(noisy, denoised)}

    # --- table
    ms, md = out["single"]["metrics"], out["diff"]["metrics"]
    print(f"\n{'=' * 64}")
    print(f"  {label}:  single (x_dev2)   vs   diff (x_dev2 - x_dev1)")
    print(f"{'=' * 64}")
    print(f"  NRR by MM (energy removed): single {ms['NRR (dB)']:+.2f} dB  |  "
          f"diff {md['NRR (dB)']:+.2f} dB   (context only)")
    print(f"  {'metric':<20}{'single':>12}{'diff':>12}   winner")
    print(f"  {'-' * 56}")
    diff_wins = single_wins = ties = 0
    for key, name, higher_better in QUALITY_METRICS:
        vs, vd = ms[key], md[key]
        if np.isnan(vs) or np.isnan(vd):
            winner = "n/a"
        elif abs(vs - vd) < 1e-9:
            winner = "tie"; ties += 1
        else:
            better_is_diff = (vd > vs) == higher_better
            winner = "diff <--" if better_is_diff else "single"
            if better_is_diff: diff_wins += 1
            else:              single_wins += 1
        print(f"  {name:<20}{fmt_val(key, vs):>12}{fmt_val(key, vd):>12}   {winner}")
    print(f"  {'-' * 56}")
    verdict = ("diff (subtraction) better" if diff_wins > single_wins else
               "single channel better"     if single_wins > diff_wins else
               "roughly equivalent")
    print(f"  >> {diff_wins} metrics favor diff, {single_wins} favor single, "
          f"{ties} tie  ->  {verdict}")

    out["verdict"] = (diff_wins, single_wins, ties, verdict)
    return out


# ---------------------------------------------------------------------------
def plot_compare(ax_row: list, res: dict, label: str, sr: int = FS) -> None:
    ns, ds = res["single"]["noisy"], res["single"]["denoised"]
    nd, dd = res["diff"]["noisy"],   res["diff"]["denoised"]
    t = np.arange(len(ds)) / sr
    n10 = min(len(ds), 10 * sr)

    # --- ax0: denoised waveforms overlaid
    ax = ax_row[0]
    ax.plot(t[:n10], ds[:n10] / (np.max(np.abs(ds)) + 1e-12), lw=0.8, color="C0",
            label="denoised (single)")
    ax.plot(t[:n10], dd[:n10] / (np.max(np.abs(dd)) + 1e-12), lw=0.8, color="C1",
            label="denoised (diff)", alpha=0.8)
    ax.set_title(f"{label} - denoised waveforms (peak-norm, first 10 s)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("amplitude"); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # --- ax1: PSD of denoised outputs (+ the two noisy inputs faded)
    ax = ax_row[1]
    for sig, col, lbl, a in [(ns, "0.6", "noisy single", 0.4),
                             (nd, "0.3", "noisy diff",   0.4),
                             (ds, "C0", "denoised single", 1.0),
                             (dd, "C1", "denoised diff",   1.0)]:
        f, psd = welch(sig, fs=sr, nperseg=1024)
        ax.semilogy(f, psd, lw=0.9, color=col, label=lbl, alpha=a)
    ax.axvspan(PCG_LO, PCG_HI, alpha=0.08, color="C0")
    ax.set_xlim(0, 400); ax.set_xlabel("freq [Hz]"); ax.set_ylabel("PSD")
    ax.set_title(f"{label} - PSD"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # --- ax2: Shannon envelopes of the two denoised outputs
    ax = ax_row[2]
    env_s = shannon_envelope(ds, sr)
    env_d = shannon_envelope(dd, sr)
    ax.plot(t[:n10], env_s[:n10], lw=1.0, color="C0", label="env denoised (single)")
    ax.plot(t[:n10], env_d[:n10], lw=1.0, color="C1", label="env denoised (diff)", alpha=0.85)
    ax.set_title(f"{label} - Shannon envelope of denoised output")
    ax.set_xlabel("time [s]"); ax.set_ylabel("Shannon energy (z)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # --- ax3: envelope autocorrelation (periodicity) of the two denoised outputs
    ax = ax_row[3]
    ac_s, lag_s, pk_s = envelope_autocorr(env_s, sr)
    ac_d, lag_d, pk_d = envelope_autocorr(env_d, sr)
    max_lag = int(2.0 * sr)
    lags = np.arange(min(max_lag, ac_s.size)) / sr
    ax.plot(lags, ac_s[:lags.size], lw=1.0, color="C0",
            label=f"single (peak {pk_s:.2f}, {60*sr/lag_s:.0f} bpm)")
    ax.plot(lags, ac_d[:lags.size], lw=1.0, color="C1",
            label=f"diff (peak {pk_d:.2f}, {60*sr/lag_d:.0f} bpm)", alpha=0.85)
    ax.axvspan(HR_LAG_LO_S, HR_LAG_HI_S, color="C0", alpha=0.05)
    ax.set_xlabel("lag [s]"); ax.set_ylabel("normalized autocorr")
    ax.set_title(f"{label} - periodicity of denoised output")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = [
        ("Talking", "data/2MIC/talking"),
        ("Walking", "data/2MIC/walking"),
    ]

    results: dict[str, dict] = {}
    for label, base_dir in sessions:
        r = compare_session(label, base_dir)
        if r is not None:
            results[label] = r

    if not results:
        print("[compare] nothing to compare -- run demo_full.py and "
              "demo_full_diff.py first.")
        return

    # --- overall verdict
    print(f"\n{'=' * 64}")
    print("  OVERALL")
    print(f"{'=' * 64}")
    tot_diff = sum(r["verdict"][0] for r in results.values())
    tot_single = sum(r["verdict"][1] for r in results.values())
    for label, r in results.items():
        dw, sw, ti, verdict = r["verdict"]
        print(f"  {label:<10}: diff {dw} / single {sw} / tie {ti}  ->  {verdict}")
    overall = ("subtraction (x_dev2 - x_dev1) helps overall" if tot_diff > tot_single else
               "single channel is better overall"            if tot_single > tot_diff else
               "no clear overall winner")
    print(f"  {'-' * 56}")
    print(f"  total: diff {tot_diff} vs single {tot_single}  ->  {overall}")

    # --- plots
    n = len(results)
    fig = plt.figure(figsize=(22, 5.5 * n))
    gs = gridspec.GridSpec(n, 4, figure=fig, hspace=0.45, wspace=0.3)
    for row, (label, r) in enumerate(results.items()):
        axes = [fig.add_subplot(gs[row, c]) for c in range(4)]
        plot_compare(axes, r, label)
    out_png = os.path.join(args.out_dir, "compare_diff.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n[compare] figure saved -> {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
