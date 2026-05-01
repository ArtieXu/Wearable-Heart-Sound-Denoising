"""
Warm-up demo for reference-guided PCG denoising.

Pipeline:
  1. Load mic_speaking.wav and mic_walking.wav.
  2. Band-pass to 25-150 Hz (the PCG band) and resample to FS_TARGET.
  3. Take a NOISY_FILE as observation y, the OTHER one as the reference z
     (warm-up only -- in production, z would be a clinically clean baseline).
  4. Phase-align z to y by cross-correlation with optimal scale.
  5. Run MM and GD solvers; plot waveforms, spectrograms, objective curves.

Usage:
    python demo.py                         # default: noisy=data/mic_walking.wav, ref=data/mic_speaking.wav
    python demo.py --noisy data/mic_speaking.wav --ref data/mic_walking.wav
    python demo.py --no-ref                # disable reference term (lambda_r=0)
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly, spectrogram

import pcg_denoise as pd


FS_TARGET = 2000           # Hz; 2 kHz is well above PCG band (<=150 Hz)
PCG_BAND = (25.0, 150.0)


# ---------------------------------------------------------------------------
def load_wav_float(path: str) -> tuple[int, np.ndarray]:
    """Read a wav file and return (sr, signal-in-[-1,1])."""
    sr, x = wavfile.read(path)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        max_abs = float(np.iinfo(x.dtype).max)
        x = x.astype(np.float64) / max_abs
    else:
        x = x.astype(np.float64)
    return sr, x


def preprocess(x: np.ndarray, sr_in: int, sr_out: int = FS_TARGET,
               band: tuple[float, float] = PCG_BAND) -> np.ndarray:
    """Band-pass to PCG band, then polyphase-resample to sr_out."""
    nyq = 0.5 * sr_in
    sos = butter(4, [band[0] / nyq, band[1] / nyq], btype="bandpass", output="sos")
    x_bp = sosfiltfilt(sos, x)

    # Rational resample sr_in -> sr_out via gcd
    from math import gcd
    g = gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    return resample_poly(x_bp, up, down)


# ---------------------------------------------------------------------------
def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """SNR in dB treating `clean` as signal and (noisy - clean) as noise."""
    s = float(np.dot(clean, clean))
    n = float(np.dot(noisy - clean, noisy - clean))
    return 10.0 * np.log10(s / max(n, 1e-20))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy", default="data/mic_walking.wav",
                    help="WAV used as the noisy observation y")
    ap.add_argument("--ref", default="data/mic_speaking.wav",
                    help="WAV used as the reference z (warm-up proxy)")
    ap.add_argument("--no-ref", action="store_true",
                    help="set lambda_r = 0 (skip reference attraction term)")
    ap.add_argument("--lambda-g", type=float, default=0.05)
    ap.add_argument("--lambda-r", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=50,
                    help="OGS group size (samples). Default 50 = 25ms at 2kHz.")
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=None,
                    help="Huber knee. Default = noise std estimate.")
    ap.add_argument("--max-sec", type=float, default=8.0,
                    help="Truncate to first N seconds for the demo.")
    ap.add_argument("--mm-iters", type=int, default=80)
    ap.add_argument("--gd-iters", type=int, default=400)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[demo] noisy = {args.noisy}")
    print(f"[demo] ref   = {args.ref}")

    # --- 1. Load
    sr_y, y_raw = load_wav_float(args.noisy)
    sr_z, z_raw = load_wav_float(args.ref)
    print(f"[demo] noisy: sr={sr_y} Hz, len={len(y_raw)} ({len(y_raw)/sr_y:.2f}s)")
    print(f"[demo] ref  : sr={sr_z} Hz, len={len(z_raw)} ({len(z_raw)/sr_z:.2f}s)")

    # --- 2. Bandpass + resample to 2 kHz
    y_full = preprocess(y_raw, sr_y, FS_TARGET, PCG_BAND)
    z_full = preprocess(z_raw, sr_z, FS_TARGET, PCG_BAND)

    n_max = int(args.max_sec * FS_TARGET)
    y = y_full[:n_max].astype(np.float64)
    n = len(y)
    print(f"[demo] working at {FS_TARGET} Hz, n={n} ({n / FS_TARGET:.2f}s)")

    # --- 3. Reference alignment
    if args.no_ref or args.lambda_r == 0.0:
        z = np.zeros(n)
        lambda_r = 0.0
        print("[demo] reference term disabled (lambda_r = 0)")
    else:
        z, tau, alpha = pd.align_reference(y, z_full, fit_scale=True)
        lambda_r = args.lambda_r
        print(f"[demo] reference aligned: tau={tau} samples, alpha={alpha:+.3f}")

    # --- 4. Hyperparameters
    noise_std = float(np.median(np.abs(y - np.median(y))) / 0.6745)
    delta = args.delta if args.delta is not None else 3.0 * noise_std
    print(f"[demo] noise_std (MAD)={noise_std:.4f}, delta={delta:.4f}")
    print(f"[demo] lambda_g={args.lambda_g}, lambda_r={lambda_r}, K={args.K}, eps={args.eps}")

    common = dict(z=z, lambda_g=args.lambda_g, lambda_r=lambda_r,
                  K=args.K, eps=args.eps, delta=delta)

    # --- 5. Solvers
    print("\n[demo] running MM ...")
    t0 = time.time()
    x_mm, hist_mm = pd.solve_mm(y, max_iter=args.mm_iters, tol=1e-8,
                                verbose=True, **common)
    t_mm = time.time() - t0
    print(f"[demo] MM: {len(hist_mm['obj'])-1} outer iters in {t_mm:.3f}s,"
          f"  F_final={hist_mm['obj'][-1]:.4e}")

    print("\n[demo] running GD (baseline) ...")
    t0 = time.time()
    x_gd, hist_gd = pd.solve_gd(y, max_iter=args.gd_iters, tol=1e-8,
                                verbose=False, **common)
    t_gd = time.time() - t0
    print(f"[demo] GD: {len(hist_gd['obj'])-1} iters in {t_gd:.3f}s,"
          f"  F_final={hist_gd['obj'][-1]:.4e}")

    # --- 6. Diagnostics
    res_mm = y - x_mm
    res_gd = y - x_gd
    print("\n[demo] energy of removed component (||y - x_hat||^2):")
    print(f"          MM = {np.dot(res_mm, res_mm):.4e}")
    print(f"          GD = {np.dot(res_gd, res_gd):.4e}")
    print("[demo] energy of denoised  (||x_hat||^2):")
    print(f"          MM = {np.dot(x_mm, x_mm):.4e}")
    print(f"          GD = {np.dot(x_gd, x_gd):.4e}")
    if lambda_r > 0:
        print(f"[demo] reference fit: ||x_mm - z|| = {np.linalg.norm(x_mm - z):.3e},"
              f"   ||y - z|| = {np.linalg.norm(y - z):.3e}")

    # --- 7. Plot waveforms + objective + spectrograms
    t = np.arange(n) / FS_TARGET
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=False)

    axes[0].plot(t, y, lw=0.8, color="0.4", label="y (noisy)")
    axes[0].plot(t, x_mm, lw=1.0, color="C0", label="x_hat (MM)")
    if lambda_r > 0:
        axes[0].plot(t, z, lw=0.8, color="C2", alpha=0.6, label="z (reference)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"Denoised PCG (MM)  --  {os.path.basename(args.noisy)}")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # GD vs MM overlay
    axes[1].plot(t, x_gd, lw=0.8, color="C1", label="x_hat (GD)")
    axes[1].plot(t, x_mm, lw=0.8, color="C0", alpha=0.7, label="x_hat (MM)")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("amplitude")
    axes[1].set_title("GD vs MM denoised solutions (should overlap if both converged)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.3)

    # objective curves
    axes[2].semilogy(np.array(hist_mm["obj"]) - min(hist_mm["obj"][-1], hist_gd["obj"][-1]) + 1e-12,
                     marker="o", ms=3, color="C0", label=f"MM ({len(hist_mm['obj'])} iters)")
    axes[2].semilogy(np.array(hist_gd["obj"]) - min(hist_mm["obj"][-1], hist_gd["obj"][-1]) + 1e-12,
                     lw=0.8, color="C1", label=f"GD ({len(hist_gd['obj'])} iters)")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel("F(x_t) - F*  +  eps")
    axes[2].set_title("Convergence (suboptimality vs. iteration)")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(loc="upper right", fontsize=8)

    # spectrogram of y vs x_mm
    f1, t1, S_y = spectrogram(y, fs=FS_TARGET, nperseg=256, noverlap=192)
    f2, t2, S_x = spectrogram(x_mm, fs=FS_TARGET, nperseg=256, noverlap=192)
    axes[3].pcolormesh(
        t1,
        f1,
        10 * np.log10(np.maximum(S_y, 1e-12)) - 10 * np.log10(np.maximum(S_x, 1e-12)),
        shading="auto",
        cmap="coolwarm",
        vmin=-20, vmax=20,
    )
    axes[3].set_xlabel("time [s]")
    axes[3].set_ylabel("freq [Hz]")
    axes[3].set_title("Spectrogram difference  (PSD_y - PSD_x_mm) [dB]   (red=removed energy)")
    axes[3].set_ylim(0, 200)

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "denoise_demo.png")
    fig.savefig(out_png, dpi=150)
    print(f"\n[demo] figure saved -> {out_png}")

    # Save denoised waveform
    out_wav = os.path.join(args.out_dir, "denoised_mm.wav")
    x_int = np.clip(x_mm / max(np.max(np.abs(x_mm)), 1e-12), -1, 1)
    wavfile.write(out_wav, FS_TARGET, (x_int * 32767).astype(np.int16))
    print(f"[demo] wav saved   -> {out_wav}")


if __name__ == "__main__":
    main()
