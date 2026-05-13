"""
Full-recording denoiser for the 2MIC talking session.

Processes every window in data/2MIC/talking/talking.npz, reconstructs a
continuous signal via overlap-add, and writes:
  data/2MIC/talking/noisy.wav        -- bandpass-filtered noisy input
  data/2MIC/talking/denoised_mm.wav  -- MM-denoised output

Usage:
    python demo_full.py
    python demo_full.py --dev 1
    python demo_full.py --lambda-g 0.03 --lambda-r 0.3
    python demo_full.py --no-ref
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

import pcg_denoise as pd


FS_TARGET = 4000
PCG_BAND  = (25.0, 150.0)
STEP_S    = 1.0          # window step in the .npz
WIN_S     = 8.0          # window length in the .npz


# ---------------------------------------------------------------------------
def load_npz_all_windows(path: str, dev: int) -> tuple[int, np.ndarray]:
    """Return (sample_rate, windows) where windows.shape = (N, win_len)."""
    data = np.load(path, allow_pickle=True)
    key  = f"x_dev{dev}"
    if key not in data.files:
        raise KeyError(f"{path} missing key {key!r}; got {data.files}")
    arr = data[key].astype(np.float64)       # (N, win_len)
    # peak-normalise each window independently
    peaks = np.max(np.abs(arr), axis=1, keepdims=True)
    peaks = np.where(peaks > 0, peaks, 1.0)
    arr   = arr / peaks
    sr    = int(float(data["meta"].item()["sample_rate_hz"]))
    return sr, arr


def load_npz_one_window(path: str, dev: int, window_idx: int) -> tuple[int, np.ndarray]:
    data  = np.load(path, allow_pickle=True)
    key   = f"x_dev{dev}"
    sig   = data[key][window_idx].astype(np.float64)
    peak  = float(np.max(np.abs(sig)))
    if peak > 0:
        sig = sig / peak
    sr = int(float(data["meta"].item()["sample_rate_hz"]))
    return sr, sig


def bandpass(x: np.ndarray, sr: int) -> np.ndarray:
    nyq = 0.5 * sr
    sos = butter(4, [PCG_BAND[0] / nyq, PCG_BAND[1] / nyq],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def overlap_add(windows: np.ndarray, step: int, total_len: int) -> np.ndarray:
    """Hann-weighted overlap-add of shape-(N, win_len) array."""
    n_win, win_len = windows.shape
    hann = np.hanning(win_len)
    out     = np.zeros(total_len)
    weights = np.zeros(total_len)
    for i, w in enumerate(windows):
        s = i * step
        e = s + win_len
        out[s:e]     += hann * w
        weights[s:e] += hann
    # avoid divide-by-zero at edges where Hann → 0
    weights = np.where(weights < 1e-12, 1.0, weights)
    return out / weights


def save_wav(path: str, sig: np.ndarray, sr: int) -> None:
    sig_norm = np.clip(sig / max(float(np.max(np.abs(sig))), 1e-12), -1, 1)
    wavfile.write(path, sr, (sig_norm * 32767).astype(np.int16))
    print(f"[full] wav saved -> {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--talking-npz", default="data/2MIC/talking/talking.npz")
    ap.add_argument("--ref-npz",     default="data/2MIC/sitting quiet/2/sit_quiet.npz")
    ap.add_argument("--dev",         type=int,   default=2, choices=(1, 2))
    ap.add_argument("--ref-window",  type=int,   default=2,
                    help="Which sit-quiet window to use as reference (default 2).")
    ap.add_argument("--no-ref",      action="store_true")
    ap.add_argument("--lambda-g",    type=float, default=0.05)
    ap.add_argument("--lambda-r",    type=float, default=0.5)
    ap.add_argument("--K",           type=int,   default=100)
    ap.add_argument("--eps",         type=float, default=1e-3)
    ap.add_argument("--delta",       type=float, default=None)
    ap.add_argument("--mm-iters",    type=int,   default=80)
    args = ap.parse_args()

    out_dir = os.path.dirname(args.talking_npz)

    # --- Load all talking windows
    sr, windows_raw = load_npz_all_windows(args.talking_npz, dev=args.dev)
    n_win, win_len  = windows_raw.shape
    step            = int(STEP_S * sr)
    total_len       = (n_win - 1) * step + win_len
    print(f"[full] loaded {n_win} windows × {win_len} samples  "
          f"→ total {total_len/sr:.1f}s  (sr={sr} Hz, dev={args.dev})")

    # --- Bandpass every talking window
    windows_bp = np.array([bandpass(w, sr) for w in windows_raw])

    # --- Bandpass the reference window
    if args.no_ref or args.lambda_r == 0.0:
        z_full   = np.zeros(win_len)
        lambda_r = 0.0
        print("[full] reference term disabled")
    else:
        sr_z, z_raw = load_npz_one_window(args.ref_npz, dev=args.dev,
                                          window_idx=args.ref_window)
        z_full   = bandpass(z_raw, sr_z)
        lambda_r = args.lambda_r
        print(f"[full] reference: {args.ref_npz}  (x_dev{args.dev}, window {args.ref_window})")

    # --- Denoise each window with MM
    windows_denoised = np.empty_like(windows_bp)
    t_start = time.time()
    for i, y in enumerate(windows_bp):
        noise_std = float(np.median(np.abs(y - np.median(y))) / 0.6745)
        delta     = args.delta if args.delta is not None else 3.0 * noise_std

        if lambda_r > 0:
            z, _, _ = pd.align_reference(y, z_full, fit_scale=True)
        else:
            z = np.zeros(win_len)

        x_mm, _ = pd.solve_mm(
            y, z=z,
            lambda_g=args.lambda_g, lambda_r=lambda_r,
            K=args.K, eps=args.eps, delta=delta,
            max_iter=args.mm_iters, tol=1e-8, verbose=False,
        )
        windows_denoised[i] = x_mm

        if (i + 1) % 50 == 0 or i == n_win - 1:
            elapsed = time.time() - t_start
            eta     = elapsed / (i + 1) * (n_win - i - 1)
            print(f"[full]  window {i+1:4d}/{n_win}   "
                  f"elapsed {elapsed:5.1f}s   ETA {eta:5.1f}s")

    print(f"[full] denoising done in {time.time()-t_start:.1f}s")

    # --- Overlap-add reconstruction
    noisy_full    = overlap_add(windows_bp,       step, total_len)
    denoised_full = overlap_add(windows_denoised, step, total_len)

    # --- Save
    save_wav(os.path.join(out_dir, "noisy.wav"),       noisy_full,    sr)
    save_wav(os.path.join(out_dir, "denoised_mm.wav"), denoised_full, sr)


if __name__ == "__main__":
    main()
