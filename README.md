# Wearable Heart-Sound Denoising

EE 596 final project — denoising phonocardiogram (PCG) recordings captured by a
wearable contact microphone, using a clean stethoscope-style baseline as a soft
reference. Talking and walking artifacts are suppressed while the S1/S2 heart
transients are preserved.

The core idea: motion and speech contaminate wearable PCG, but the *shape* of
the heart sound (sharp S1/S2 transients on a quiet background) is well predicted
by a phase-aligned baseline recorded in a quiet posture. We exploit that with a
smooth composite objective and solve it with two solvers — gradient descent and
a custom majorization–minimization (MM) scheme that turns each outer step into
an O(n) tridiagonal solve.

## Method

Estimate the clean PCG `x` from the noisy observation `y` by minimizing

```
F(x) = 0.5 * ||x - y||^2
     + lambda_g * sum_g sqrt( ||(D x)[g : g+K-1]||^2 + eps^2 )
     + lambda_r * sum_i  w_i * huber_delta(x_i - z_i)
```

- **Data fidelity** `0.5 ||x - y||^2` keeps `x` close to the observation.
- **Overlapping group sparsity (OGS)** on the first-difference `D x` promotes
  *piecewise-smooth* segments separated by sharp S1/S2 transients
  (Selesnick & Chen 2013). The `eps`-smoothing makes the penalty differentiable.
- **Robust reference attraction** pulls `x` toward a phase-aligned quiet
  baseline `z`. The Huber loss caps the influence of large mismatches so motion
  artifacts in `y` cannot drag `x` toward an arbitrary reference.

### Solvers

Both solvers live in [pcg_denoise.py](pcg_denoise.py):

- `solve_gd` — monotone gradient descent with Armijo backtracking.
- `solve_mm` — majorization–minimization. Each outer step builds a quadratic
  upper bound (linear majorizer for the concave OGS-in-energy term, half-
  quadratic majorizer for the Huber loss) and minimizes it via a symmetric
  tridiagonal solve in **O(n)** with `scipy.linalg.solveh_banded`. Monotone
  non-increasing by construction.

A finite-difference gradient check (`check_gradient`) verifies the analytical
gradient before tuning hyperparameters.

## Repository layout

```
.
├── pcg_denoise.py            # core algorithm: objective, gradient, solve_gd, solve_mm
├── demo.py                   # single-window demo (one 8-s window, MM vs GD, plots)
├── demo_full.py              # full-session denoiser on x_dev2 (overlap-add)
├── demo_full_diff.py         # full-session denoiser on x_dev2 - x_dev1 (diff channel)
├── analyze.py                # reference-free PCG quality metrics + plots
├── compare_diff.py           # head-to-head: single channel vs diff channel
├── requirements.txt
├── LICENSE
└── results/
    └── analysis/
        ├── analysis.png      # per-session metric/plots panel
        ├── compare_diff.png  # single vs diff comparison panel
        ├── talking_results/talking/   # talking-session wavs (noisy + denoised, single & diff)
        └── walking_results/walking/   # walking-session wavs (noisy + denoised, single & diff)
```

## Data

The raw recordings and intermediate `.npz` window arrays are **not committed**
(too large for git). The pipeline expects this on-disk layout:

```
data/2MIC/
├── sitting quiet/
│   ├── 1/sit_quiet.npz
│   └── 2/sit_quiet.npz       # used as the quiet reference
├── talking/talking.npz
└── walking/walking.npz
```

Each `.npz` contains keyed arrays `x_dev1`, `x_dev2` of shape `(n_windows,
win_len)` and a `meta` dict with `sample_rate_hz` (the 2MIC sessions are at 4
kHz, 8-s windows, 1-s step).

Pre-rendered denoised WAVs from one full pass over the talking and walking
sessions are included under `results/analysis/{talking_results,walking_results}/`
so the analysis scripts can be re-run without the raw data.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### 1. Single-window demo (no overlap-add)

Default: one 8-s talking window denoised against one quiet reference window,
with MM and GD overlaid, convergence curves, and the spectrogram difference.

```bash
python demo.py
# pick a different window / device channel
python demo.py --window-idx 50 --dev 1
# disable the reference term (pure OGS denoising)
python demo.py --no-ref
```

Writes `results/denoise_demo.png` and `results/denoised_mm.wav`.

### 2. Full-session denoising (overlap-add)

Process every window in a session `.npz`, MM-denoise each window, and Hann-
overlap-add into a continuous signal. Two front-ends:

```bash
# Single channel (y = x_dev2)
python demo_full.py --talking-npz data/2MIC/talking/talking.npz
python demo_full.py --talking-npz data/2MIC/walking/walking.npz

# Differential channel (y = x_dev2 - x_dev1) — cancels common-mode noise
python demo_full_diff.py --talking-npz data/2MIC/talking/talking.npz
python demo_full_diff.py --talking-npz data/2MIC/walking/walking.npz
```

Each call writes `noisy.wav` / `denoised_mm.wav` (or the `_diff` variants) next
to the input `.npz`.

### 3. Reference-free quality analysis

There is no clean ground-truth heart sound for wearable recordings, so standard
supervised metrics (SNR-improvement against a known clean signal) don't apply.
`analyze.py` reports the *reference-free* PCG quality measures from the heart-
sound signal-quality-assessment literature:

- **NRR** — noise reduction ratio (energy removed by the denoiser).
- **SNR_env** — segment-based SNR using a Shannon-energy envelope to split
  active (S1/S2) vs silent (systole/diastole) samples.
- **SNR_pk** — peak-to-noise SNR à la Grimaldi et al., *Sensors* 2021;
  ≥14 dB is the clinical usability threshold.
- **SNR_band** — PSD power inside the 25–150 Hz PCG band vs above it.
- **Periodicity SQI** — peak of the envelope autocorrelation in the cardiac-lag
  range (40–200 bpm); also yields an implied HR.
- **Kurtosis** and **spectral energy concentration** inside the PCG band.

```bash
python analyze.py
```

Reads `data/2MIC/{talking,walking}/{noisy,denoised_mm}.wav`, prints a per-
session metrics table, and writes `results/analysis/analysis.png`.

### 4. Single vs differential front-end

Does subtracting the environmental-noise mic (`x_dev2 - x_dev1`) before MM
denoising actually help? Run both `demo_full.py` and `demo_full_diff.py` on the
same session, then:

```bash
python compare_diff.py
```

Scores each front-end on the six denoised-output quality metrics, prints
per-metric winners, and writes `results/analysis/compare_diff.png`.

## Sanity checks

Run `pcg_denoise.py` directly to verify the analytic gradient against central
differences and to confirm MM is monotone non-increasing on a synthetic signal:

```bash
python pcg_denoise.py
```

## Hyperparameters at a glance

| Symbol      | Meaning                                      | Default      |
| ----------- | -------------------------------------------- | ------------ |
| `lambda_g`  | OGS group-sparsity weight on `D x`           | `0.05`       |
| `lambda_r`  | Reference-attraction weight                  | `0.5`        |
| `K`         | OGS group size in samples (≈25 ms @ 4 kHz)   | `100`        |
| `eps`       | OGS smoothing constant                       | `1e-3`       |
| `delta`     | Huber knee                                   | `3 × MAD(y)` |
| `FS_TARGET` | Working sample rate (2MIC native)            | `4000` Hz    |
| `PCG_BAND`  | Pre-bandpass for the PCG band                | `25–150` Hz  |

## References

- Selesnick & Chen, *Total variation denoising with overlapping group sparsity*,
  ICASSP 2013.
- Deng & Han, *Adaptive overlapping-group sparse denoising for heart sound
  signals*, 2018.
- Huber, *Robust estimation of a location parameter*, Ann. Math. Stat. 1964.
- Beck & Teboulle, *A fast iterative shrinkage-thresholding algorithm (FISTA)*,
  SIAM J. Imaging Sci. 2009.
- O'Donoghue & Candès, *Adaptive restart for accelerated gradient schemes*,
  Found. Comput. Math. 2015.
- Liang, Lukkarinen, Hartimo, *Heart sound segmentation algorithm based on
  heart sound envelogram*, Computers in Cardiology 1997.
- Grimaldi et al., *Automated assessment of the quality of phonocardiographic
  recordings through the SNR for home monitoring applications*, Sensors 21(21):
  7246, 2021.
- Tang et al., *Automated signal quality assessment for heart sound signal by
  novel features and evaluation in open public datasets*, BioMed Res. Int. 2021.
- Mishra et al., *Evaluation of performance metrics and denoising of PCG signal
  using wavelet-based decomposition*, IEEE 2020.

## License

[MIT](LICENSE) © Zechen Xu
