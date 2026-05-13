"""
Full-recording denoiser using the DIFFERENTIAL channel  x_dev2 - x_dev1.

Same pipeline as demo_full.py, but the "noisy" observation y for every window
is the difference of the two stethoscope channels instead of a single mic.
Subtracting the two contact mics cancels common-mode body / handling noise that
both pick up, leaving (mostly) the differential heart-sound component.

For each 8-s window:
  1. y_diff = x_dev2 - x_dev1   (then peak-normalized)
  2. band-pass to 25-150 Hz
  3. align the quiet reference (also a x_dev2 - x_dev1 window) to y
  4. MM denoise
  5. Hann overlap-add into a continuous signal

Writes (next to the input .npz):
  noisy_diff.wav        -- bandpass-filtered x_dev2 - x_dev1
  denoised_mm_diff.wav  -- MM-denoised output

Usage:
    python demo_full_diff.py
    python demo_full_diff.py --talking-npz data/2MIC/walking/walking.npz
    python demo_full_diff.py --lambda-g 0.03 --lambda-r 0.3
    python demo_full_diff.py --no-ref
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
def _peak_norm(arr: np.ndarray) -> np.ndarray:
    """Peak-normalize along the last axis (per window)."""
    peaks = np.max(np.abs(arr), axis=-1, keepdims=True)
    peaks = np.where(peaks > 0, peaks, 1.0)
    return arr / peaks


def load_diff_all_windows(path: str) -> tuple[int, np.ndarray]:
    """Return (sample_rate, windows) with windows = peak_norm(x_dev2 - x_dev1)."""
    data = np.load(path, allow_pickle=True)
    for key in ("x_dev1", "x_dev2"):
        if key not in data.files:
            raise KeyError(f"{path} missing key {key!r}; got {data.files}")
    diff = data["x_dev2"].astype(np.float64) - data["x_dev1"].astype(np.float64)
    sr   = int(float(data["meta"].item()["sample_rate_hz"]))
    return sr, _peak_norm(diff)


def load_diff_one_window(path: str, window_idx: int) -> tuple[int, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    diff = (data["x_dev2"][window_idx].astype(np.float64)
            - data["x_dev1"][window_idx].astype(np.float64))
    sr   = int(float(data["meta"].item()["sample_rate_hz"]))
    return sr, _peak_norm(diff)


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
    weights = np.where(weights < 1e-12, 1.0, weights)
    return out / weights


def save_wav(path: str, sig: np.ndarray, sr: int) -> None:
    sig_norm = np.clip(sig / max(float(np.max(np.abs(sig))), 1e-12), -1, 1)
    wavfile.write(path, sr, (sig_norm * 32767).astype(np.int16))
    print(f"[diff] wav saved -> {path}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--talking-npz", default="data/2MIC/talking/talking.npz",
                    help="Noisy session .npz (any 2MIC session works).")
    ap.add_argument("--ref-npz",     default="data/2MIC/sitting quiet/2/sit_quiet.npz")
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

    # --- Load all windows as the differential channel
    sr, windows_raw = load_diff_all_windows(args.talking_npz)
    n_win, win_len  = windows_raw.shape
    step            = int(STEP_S * sr)
    total_len       = (n_win - 1) * step + win_len
    print(f"[diff] input = x_dev2 - x_dev1   ({args.talking_npz})")
    print(f"[diff] loaded {n_win} windows × {win_len} samples  "
          f"→ total {total_len/sr:.1f}s  (sr={sr} Hz)")

    # --- Bandpass every window
    windows_bp = np.array([bandpass(w, sr) for w in windows_raw])

    # --- Bandpass the reference window (also differential)
    if args.no_ref or args.lambda_r == 0.0:
        z_full   = np.zeros(win_len)
        lambda_r = 0.0
        print("[diff] reference term disabled")
    else:
        sr_z, z_raw = load_diff_one_window(args.ref_npz, window_idx=args.ref_window)
        z_full   = bandpass(z_raw, sr_z)
        lambda_r = args.lambda_r
        print(f"[diff] reference: {args.ref_npz}  (x_dev2 - x_dev1, window {args.ref_window})")

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
            print(f"[diff]  window {i+1:4d}/{n_win}   "
                  f"elapsed {elapsed:5.1f}s   ETA {eta:5.1f}s")

    print(f"[diff] denoising done in {time.time()-t_start:.1f}s")

    # --- Overlap-add reconstruction
    noisy_full    = overlap_add(windows_bp,       step, total_len)
    denoised_full = overlap_add(windows_denoised, step, total_len)

    # --- Save
    save_wav(os.path.join(out_dir, "noisy_diff.wav"),       noisy_full,    sr)
    save_wav(os.path.join(out_dir, "denoised_mm_diff.wav"), denoised_full, sr)


if __name__ == "__main__":
    main()
