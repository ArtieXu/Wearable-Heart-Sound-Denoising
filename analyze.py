"""
Reference-free evaluation of PCG denoising results.

We have no clean ground-truth heart sound for the wearable recordings, so the
standard supervised metrics (output-SNR / SNR-improvement / MSE / PRD against a
known clean signal — used e.g. in wavelet PCG denoising studies such as
Mishra et al., "Evaluation of Performance Metrics and Denoising of PCG Signal
using Wavelet Based Decomposition", IEEE 2020, and the ECG/PCG wavelet-denoising
literature) cannot be computed directly.  Instead we adopt the *reference-free*
quality measures used in the heart-sound signal-quality-assessment literature:

  1. Noise Reduction Ratio (NRR) and RMS reduction
     NRR = 10 log10( E[y^2] / E[(y-x_hat)^2] ) — energy removed by the denoiser.
     (the unsupervised counterpart of "SNR improvement" reported in the wavelet
      PCG/ECG denoising papers cited above.)

  2. Envelope-segmented SNR (SNR_env)
     Partition the recording into "heart-sound active" segments (S1/S2 — high
     Shannon-energy envelope) and "silent" segments (systolic/diastolic pauses),
     then
        SNR_env = 10 log10( mean power in active segments
                            / mean power in silent segments ).
     Mirrors the segment-based SNR / energy-ratio SQIs of the PhysioNet/CinC
     2016 challenge feature set and Tang et al., "Automated Signal Quality
     Assessment for Heart Sound Signal by Novel Features...", BioMed Res. Int.
     2021.

  3. Peak-to-noise SNR (SNR_pk)
     SNR_pk = 20 log10( A_S1S2 / (4 * sigma_silent) ), where A_S1S2 is the
     median peak-to-peak amplitude of detected S1/S2 complexes and sigma_silent
     the std of the quietest cardiac-cycle intervals.  This is the clinical PCG
     SNR of Grimaldi et al., "Automated Assessment of the Quality of
     Phonocardiographic Recordings through the SNR for Home Monitoring
     Applications", Sensors 21(21):7246, 2021 (acceptability threshold ~14 dB).

  4. Band SNR (SNR_band)
     SNR_band = 10 log10( PSD power in the 25-150 Hz PCG band / PSD power above
     it ).  A denoiser that strips motion/speech without eating the heart sound
     should raise this. (high-frequency energy-ratio SQI; PhysioNet 2016 / Tang
     2021.)

  5. Heartbeat periodicity SQI
     Height of the largest peak of the normalized envelope autocorrelation in
     the cardiac-lag range (40-200 bpm).  A clean PCG is quasi-periodic;
     artifacts destroy that, so denoising should raise the peak. (degree-of-
     periodicity / autocorrelation-peak SQIs, PhysioNet 2016.)  We also report
     the implied heart rate.

  6. Kurtosis & spectral energy concentration
     Impulsiveness of S1/S2 and fraction of spectral energy inside the PCG band,
     before/after. (kurtosis SQI, PhysioNet 2016.)

Plots: waveforms, PSD, spectrogram difference, window-RMS, the Shannon envelope
with detected heart-sound/silence segments, and the envelope autocorrelation
(noisy vs denoised).

Usage:
    python analyze.py
    python analyze.py --out-dir results/analysis
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
from scipy.signal import welch, spectrogram
from scipy.stats import kurtosis


FS = 4000
PCG_LO, PCG_HI = 25.0, 150.0          # PCG band
HR_LAG_LO_S, HR_LAG_HI_S = 0.30, 1.5  # cardiac period range -> 40-200 bpm


# ---------------------------------------------------------------------------
def load_wav(path: str) -> np.ndarray:
    sr, data = wavfile.read(path)
    assert sr == FS, f"expected {FS} Hz, got {sr}"
    return data.astype(np.float64) / 32767.0


# --- core unsupervised metrics ---------------------------------------------
def nrr_db(noisy: np.ndarray, denoised: np.ndarray) -> float:
    """Noise Reduction Ratio: 10*log10(E[y^2] / E[residual^2])."""
    residual = noisy - denoised
    return 10.0 * np.log10(np.dot(noisy, noisy) / max(np.dot(residual, residual), 1e-20))


def spectral_concentration(sig: np.ndarray, sr: int,
                           lo: float = PCG_LO, hi: float = PCG_HI) -> float:
    """Fraction of PSD energy inside [lo, hi] Hz."""
    f, psd = welch(sig, fs=sr, nperseg=512)
    in_band = psd[(f >= lo) & (f <= hi)].sum()
    total = psd.sum()
    return float(in_band / max(total, 1e-20))


def band_snr_db(sig: np.ndarray, sr: int,
                lo: float = PCG_LO, hi: float = PCG_HI) -> float:
    """10*log10( PSD power in [lo,hi] / PSD power in (hi, Nyquist] )."""
    f, psd = welch(sig, fs=sr, nperseg=1024)
    in_band = psd[(f >= lo) & (f <= hi)].sum()
    out_band = psd[f > hi].sum()
    return 10.0 * np.log10(in_band / max(out_band, 1e-20))


# --- envelope / segmentation -----------------------------------------------
def shannon_envelope(sig: np.ndarray, sr: int, smooth_ms: float = 50.0) -> np.ndarray:
    """Normalized average Shannon-energy envelope, z-scored.

    Shannon energy -x^2 * log(x^2) emphasises mid-amplitude content (S1/S2)
    over both silence and impulsive spikes; a moving-average over ~50 ms gives
    a smooth envelope. (Liang et al. 1997; standard PCG envelope.)
    """
    x = sig / (np.max(np.abs(sig)) + 1e-12)
    e = -(x ** 2) * np.log(x ** 2 + 1e-12)
    k = max(1, int(smooth_ms * 1e-3 * sr))
    kern = np.ones(k) / k
    env = np.convolve(e, kern, mode="same")
    return (env - env.mean()) / (env.std() + 1e-12)


def segment_masks(env: np.ndarray, active_pct: float = 75.0,
                  silent_pct: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for heart-sound-active and silent samples from envelope.

    'active'  = top (100 - active_pct)% of envelope values  (S1/S2 complexes),
    'silent'  = bottom silent_pct% of envelope values       (systole/diastole).
    """
    hi = np.percentile(env, active_pct)
    lo = np.percentile(env, silent_pct)
    return env >= hi, env <= lo


def connected_segments(mask: np.ndarray) -> list[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return np.split(idx, splits)


def envelope_snr_db(sig: np.ndarray, active: np.ndarray, silent: np.ndarray) -> float:
    if active.sum() == 0 or silent.sum() == 0:
        return float("nan")
    p_act = np.mean(sig[active] ** 2)
    p_sil = np.mean(sig[silent] ** 2)
    return 10.0 * np.log10(p_act / max(p_sil, 1e-20))


def peak_to_noise_snr_db(sig: np.ndarray, active: np.ndarray, silent: np.ndarray,
                         sr: int, min_seg_ms: float = 20.0) -> float:
    """20*log10( median peak-to-peak of S1/S2 complexes / (4*sigma_silent) ).

    Grimaldi et al., Sensors 2021.
    """
    if silent.sum() == 0:
        return float("nan")
    sigma = np.std(sig[silent])
    min_len = int(min_seg_ms * 1e-3 * sr)
    pps = [np.ptp(sig[seg]) for seg in connected_segments(active) if seg.size >= min_len]
    if not pps or sigma <= 0:
        return float("nan")
    return 20.0 * np.log10(np.median(pps) / (4.0 * sigma))


def envelope_autocorr(env: np.ndarray, sr: int) -> tuple[np.ndarray, int, float]:
    """Return (normalized one-sided autocorr, dominant cardiac lag [samples],
    autocorr peak height at that lag in [0,1]).  FFT-based: O(n log n)."""
    e = env - env.mean()
    n = e.size
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    E = np.fft.rfft(e, nfft)
    ac = np.fft.irfft(E * np.conj(E), nfft)[:n]
    ac = ac / (ac[0] + 1e-12)
    lo = int(HR_LAG_LO_S * sr)
    hi = min(ac.size - 1, int(HR_LAG_HI_S * sr))
    if hi <= lo:
        return ac, lo, float(ac[lo])
    lag = lo + int(np.argmax(ac[lo:hi]))
    return ac, lag, float(ac[lag])


# ---------------------------------------------------------------------------
def compute_metrics(noisy: np.ndarray, denoised: np.ndarray, sr: int = FS) -> dict:
    residual = noisy - denoised

    # detect heart-sound timing on the (cleaner) denoised signal, then apply the
    # SAME masks to both signals so the comparison is at identical time samples.
    env_d = shannon_envelope(denoised, sr)
    env_n = shannon_envelope(noisy, sr)
    active, silent = segment_masks(env_d)

    _, lag_n, ac_pk_n = envelope_autocorr(env_n, sr)
    _, lag_d, ac_pk_d = envelope_autocorr(env_d, sr)

    return {
        "NRR (dB)":              nrr_db(noisy, denoised),
        "RMS noisy":             float(np.sqrt(np.mean(noisy ** 2))),
        "RMS denoised":          float(np.sqrt(np.mean(denoised ** 2))),
        "RMS residual":          float(np.sqrt(np.mean(residual ** 2))),
        "SNR_env noisy (dB)":    envelope_snr_db(noisy, active, silent),
        "SNR_env denoised (dB)": envelope_snr_db(denoised, active, silent),
        "SNR_pk noisy (dB)":     peak_to_noise_snr_db(noisy, active, silent, sr),
        "SNR_pk denoised (dB)":  peak_to_noise_snr_db(denoised, active, silent, sr),
        "SNR_band noisy (dB)":   band_snr_db(noisy, sr),
        "SNR_band denoised (dB)": band_snr_db(denoised, sr),
        "periodicity noisy":     ac_pk_n,
        "periodicity denoised":  ac_pk_d,
        "HR noisy (bpm)":        60.0 * sr / lag_n,
        "HR denoised (bpm)":     60.0 * sr / lag_d,
        "kurtosis noisy":        float(kurtosis(noisy)),
        "kurtosis denoised":     float(kurtosis(denoised)),
        "kurtosis residual":     float(kurtosis(residual)),
        "spectral conc noisy":   spectral_concentration(noisy, sr),
        "spectral conc denoised": spectral_concentration(denoised, sr),
    }


def _arrow(after: float, before: float, good_up: bool = True) -> str:
    if np.isnan(after) or np.isnan(before):
        return " "
    up = after > before
    return "OK" if up == good_up else ".."


def print_metrics(label: str, m: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  NRR (energy removed)      : {m['NRR (dB)']:+6.2f} dB")
    print(f"  RMS  noisy / denoi / res  : {m['RMS noisy']:.4f} / "
          f"{m['RMS denoised']:.4f} / {m['RMS residual']:.4f}")
    print(f"  SNR_env  noisy -> denoised: {m['SNR_env noisy (dB)']:+6.2f} -> "
          f"{m['SNR_env denoised (dB)']:+6.2f} dB  "
          f"[{_arrow(m['SNR_env denoised (dB)'], m['SNR_env noisy (dB)'])}]")
    print(f"  SNR_pk   noisy -> denoised: {m['SNR_pk noisy (dB)']:+6.2f} -> "
          f"{m['SNR_pk denoised (dB)']:+6.2f} dB  "
          f"[{_arrow(m['SNR_pk denoised (dB)'], m['SNR_pk noisy (dB)'])}]  "
          f"(>=14 dB = clinically usable, Grimaldi 2021)")
    print(f"  SNR_band noisy -> denoised: {m['SNR_band noisy (dB)']:+6.2f} -> "
          f"{m['SNR_band denoised (dB)']:+6.2f} dB  "
          f"[{_arrow(m['SNR_band denoised (dB)'], m['SNR_band noisy (dB)'])}]")
    print(f"  periodicity SQI           : {m['periodicity noisy']:.3f} -> "
          f"{m['periodicity denoised']:.3f}  "
          f"[{_arrow(m['periodicity denoised'], m['periodicity noisy'])}]   "
          f"(HR ~ {m['HR noisy (bpm)']:.0f} -> {m['HR denoised (bpm)']:.0f} bpm)")
    print(f"  kurtosis noisy -> denoised: {m['kurtosis noisy']:.2f} -> "
          f"{m['kurtosis denoised']:.2f}  "
          f"[{_arrow(m['kurtosis denoised'], m['kurtosis noisy'])}]   "
          f"(residual {m['kurtosis residual']:.2f})")
    print(f"  spectral conc in PCG band : {m['spectral conc noisy'] * 100:.1f}% -> "
          f"{m['spectral conc denoised'] * 100:.1f}%  "
          f"[{_arrow(m['spectral conc denoised'], m['spectral conc noisy'])}]")


# ---------------------------------------------------------------------------
def plot_session(ax_row: list, noisy: np.ndarray, denoised: np.ndarray,
                 label: str, sr: int = FS) -> None:
    residual = noisy - denoised
    t = np.arange(len(noisy)) / sr

    # --- ax0: waveform (first 10 s)
    ax = ax_row[0]
    n10 = min(len(noisy), 10 * sr)
    ax.plot(t[:n10], noisy[:n10],    lw=0.6, color="0.5", label="noisy", alpha=0.8)
    ax.plot(t[:n10], denoised[:n10], lw=0.8, color="C0",  label="denoised")
    ax.plot(t[:n10], residual[:n10], lw=0.6, color="C3",  label="residual", alpha=0.7)
    ax.set_title(f"{label} - waveform (first 10 s)")
    ax.set_xlabel("time [s]"); ax.set_ylabel("amplitude")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

    # --- ax1: PSD
    ax = ax_row[1]
    for sig, col, lbl in [(noisy, "0.5", "noisy"),
                          (denoised, "C0", "denoised"),
                          (residual, "C3", "residual")]:
        f, psd = welch(sig, fs=sr, nperseg=1024)
        ax.semilogy(f, psd, lw=0.9, color=col, label=lbl, alpha=0.85)
    ax.axvspan(PCG_LO, PCG_HI, alpha=0.08, color="C0", label="PCG band")
    ax.set_xlim(0, 400); ax.set_xlabel("freq [Hz]"); ax.set_ylabel("PSD")
    ax.set_title(f"{label} - PSD"); ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")

    # --- ax2: spectrogram difference (noisy - denoised in dB)
    ax = ax_row[2]
    _, _, S_n = spectrogram(noisy,    fs=sr, nperseg=512, noverlap=384)
    _, _, S_d = spectrogram(denoised, fs=sr, nperseg=512, noverlap=384)
    f_s = np.linspace(0, sr / 2, S_n.shape[0])
    t_s = np.linspace(0, len(noisy) / sr, S_n.shape[1])
    diff_db = (10 * np.log10(np.maximum(S_n, 1e-14))
               - 10 * np.log10(np.maximum(S_d, 1e-14)))
    im = ax.pcolormesh(t_s, f_s, diff_db, shading="auto", cmap="coolwarm",
                       vmin=-20, vmax=20)
    plt.colorbar(im, ax=ax, label="dB removed")
    ax.set_ylim(0, 400); ax.set_xlabel("time [s]"); ax.set_ylabel("freq [Hz]")
    ax.set_title(f"{label} - spectrogram diff (red = removed energy)")

    # --- ax3: window-level RMS
    ax = ax_row[3]
    win, stp = 4 * sr, sr
    n_wins = (len(noisy) - win) // stp + 1
    rms = lambda s: np.array([np.sqrt(np.mean(s[i * stp:i * stp + win] ** 2))
                              for i in range(n_wins)])
    rms_n, rms_d = rms(noisy), rms(denoised)
    t_w = np.arange(n_wins) * stp / sr
    ax.plot(t_w, rms_n, lw=0.9, color="0.5", label="noisy")
    ax.plot(t_w, rms_d, lw=0.9, color="C0",  label="denoised")
    ax.fill_between(t_w, rms_d, rms_n, alpha=0.2, color="C3", label="removed")
    ax.set_xlabel("time [s]"); ax.set_ylabel("RMS")
    ax.set_title(f"{label} - window RMS (4 s, 1 s step)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # --- ax4: Shannon envelope + detected heart-sound / silence segments
    ax = ax_row[4]
    env_n = shannon_envelope(noisy, sr)
    env_d = shannon_envelope(denoised, sr)
    active, silent = segment_masks(env_d)
    n10 = min(len(noisy), 10 * sr)
    tt = t[:n10]
    ax.plot(tt, env_n[:n10], lw=0.7, color="0.5", label="env(noisy)", alpha=0.8)
    ax.plot(tt, env_d[:n10], lw=1.0, color="C0",  label="env(denoised)")
    ax.fill_between(tt, env_d[:n10].min(), env_d[:n10].max(),
                    where=active[:n10], color="C2", alpha=0.18, label="S1/S2 (active)")
    ax.fill_between(tt, env_d[:n10].min(), env_d[:n10].max(),
                    where=silent[:n10], color="C3", alpha=0.12, label="silent")
    ax.set_xlabel("time [s]"); ax.set_ylabel("Shannon energy (z)")
    ax.set_title(f"{label} - envelope & heart-sound segmentation")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.3)

    # --- ax5: envelope autocorrelation (periodicity SQI)
    ax = ax_row[5]
    ac_n, lag_n, pk_n = envelope_autocorr(env_n, sr)
    ac_d, lag_d, pk_d = envelope_autocorr(env_d, sr)
    max_lag = int(2.0 * sr)
    lags = np.arange(min(max_lag, ac_n.size)) / sr
    ax.plot(lags, ac_n[:lags.size], lw=0.8, color="0.5", label=f"noisy (peak {pk_n:.2f})")
    ax.plot(lags, ac_d[:lags.size], lw=1.0, color="C0",  label=f"denoised (peak {pk_d:.2f})")
    ax.axvline(lag_d / sr, color="C2", ls="--", lw=0.8,
               label=f"cardiac lag ~ {lag_d / sr:.2f}s ({60 * sr / lag_d:.0f} bpm)")
    ax.axvspan(HR_LAG_LO_S, HR_LAG_HI_S, color="C0", alpha=0.05)
    ax.set_xlabel("lag [s]"); ax.set_ylabel("normalized autocorr")
    ax.set_title(f"{label} - envelope autocorrelation (periodicity)")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sessions = [
        ("Talking", "data/2MIC/talking/noisy.wav", "data/2MIC/talking/denoised_mm.wav"),
        ("Walking", "data/2MIC/walking/noisy.wav", "data/2MIC/walking/denoised_mm.wav"),
    ]

    all_metrics: dict[str, dict] = {}
    for label, noisy_path, denoised_path in sessions:
        if not (os.path.exists(noisy_path) and os.path.exists(denoised_path)):
            print(f"[analyze] skipping {label}: missing {noisy_path} or {denoised_path}")
            continue
        noisy = load_wav(noisy_path)
        denoised = load_wav(denoised_path)
        all_metrics[label] = compute_metrics(noisy, denoised)
        print_metrics(label, all_metrics[label])

    if not all_metrics:
        print("[analyze] nothing to do (run demo_full.py first to produce the wavs).")
        return

    # --- Summary comparison table
    print(f"\n{'=' * 60}")
    print("  Summary  (n = noisy, d = denoised)")
    print(f"{'=' * 60}")
    cols = list(all_metrics.keys())
    print(f"  {'Metric':<26}" + "".join(f"{c:>14}" for c in cols))
    print(f"  {'-' * (26 + 14 * len(cols))}")
    rows = [
        ("NRR [dB]",                 ["NRR (dB)"],                                "{:+.2f}"),
        ("SNR_env n->d [dB]",        ["SNR_env noisy (dB)", "SNR_env denoised (dB)"], "{:+.1f}->{:+.1f}"),
        ("SNR_pk  n->d [dB]",        ["SNR_pk noisy (dB)", "SNR_pk denoised (dB)"],   "{:+.1f}->{:+.1f}"),
        ("SNR_band n->d [dB]",       ["SNR_band noisy (dB)", "SNR_band denoised (dB)"], "{:+.1f}->{:+.1f}"),
        ("periodicity n->d",         ["periodicity noisy", "periodicity denoised"],  "{:.2f}->{:.2f}"),
        ("kurtosis n->d",            ["kurtosis noisy", "kurtosis denoised"],        "{:.1f}->{:.1f}"),
        ("spectral conc n->d",       ["spectral conc noisy", "spectral conc denoised"], "{:.0%}->{:.0%}"),
    ]
    for name, keys, fmt in rows:
        cells = []
        for c in cols:
            vals = [all_metrics[c][k] for k in keys]
            cells.append(fmt.format(*vals))
        print(f"  {name:<26}" + "".join(f"{cell:>14}" for cell in cells))

    # --- Plots
    n_sessions = len(all_metrics)
    fig = plt.figure(figsize=(30, 6.5 * n_sessions))
    gs = gridspec.GridSpec(n_sessions, 6, figure=fig, hspace=0.45, wspace=0.32)
    for row, label in enumerate(all_metrics):
        noisy_path, denoised_path = dict((l, (n, d)) for l, n, d in sessions)[label]
        noisy = load_wav(noisy_path)
        denoised = load_wav(denoised_path)
        axes = [fig.add_subplot(gs[row, col]) for col in range(6)]
        plot_session(axes, noisy, denoised, label)

    out_png = os.path.join(args.out_dir, "analysis.png")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\n[analyze] figure saved -> {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
