from typing import Dict, Any, List
import numpy as np
import librosa
import mir_eval
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from utils import _safe_float

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 16000
FMIN = 55.0
FMAX = 1760.0
HOP_LENGTH = 512
FRAME_LENGTH = 2048
MAX_DURATION = 60.0

COMMON_GRID_FPS = 25.0
CONTOUR_GRID_FPS = 25.0
MOTIF_GRID_FPS = 10.0

MEDIAN_FILTER_WIDTH = 5
MIN_COMMON_GRID_POINTS = 2
MIN_CONTOUR_VALID_POINTS = 3
MIN_CONTOUR_DIFF_POINTS = 2
MIN_VOICED_RUN = 4

SEMITONE_IN_CENTS = 100.0
REFERENCE_FREQUENCY_HZ = 440.0
EPSILON = 1e-8

MOTIF_NGRAM_N = 3
PITCH_SHIFT_SEMITONE_DIVISOR = 12.0

# =========================================================
# F0 EXTRACTION
# =========================================================

def f0_librosa(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    fmin: float = FMIN,
    fmax: float = FMAX,
    hop_length: int = HOP_LENGTH,
    max_duration: float | None = MAX_DURATION,
):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if max_duration is not None:
        max_len = int(sr * max_duration)
        if len(y) > max_len:
            y = y[:max_len]

    y = np.nan_to_num(y, nan=0.0)

    if len(y) == 0 or np.max(np.abs(y)) < EPSILON:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

    try:
        f0, vflag, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            frame_length=FRAME_LENGTH,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        return times, np.asarray(f0, dtype=float), np.asarray(vflag, dtype=bool)
    except Exception:
        pass

    try:
        f0 = librosa.yin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            hop_length=hop_length,
            frame_length=FRAME_LENGTH,
        )
        times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
        f0 = np.asarray(f0, dtype=float)
        vflag = np.isfinite(f0) & (f0 >= fmin) & (f0 <= fmax)
        f0 = f0.copy()
        f0[~vflag] = np.nan
        return times, f0, vflag
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

# =========================================================
# INTERNAL HELPERS
# =========================================================


def _hz_to_cents(f: np.ndarray, tonic: float = REFERENCE_FREQUENCY_HZ) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    out = np.full_like(f, np.nan, dtype=float)
    valid = np.isfinite(f) & (f > 0)
    out[valid] = 1200.0 * np.log2(f[valid] / tonic)
    return out


def _interp_masked(times_src, values_src, times_dst, kind="linear"):
    times_src = np.asarray(times_src, dtype=float)
    values_src = np.asarray(values_src, dtype=float)
    times_dst = np.asarray(times_dst, dtype=float)

    valid = np.isfinite(times_src) & np.isfinite(values_src)
    if valid.sum() < MIN_COMMON_GRID_POINTS:
        return np.full_like(times_dst, np.nan, dtype=float)

    f = interp1d(
        times_src[valid],
        values_src[valid],
        kind=kind,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return np.asarray(f(times_dst), dtype=float)


def _interp_bool(times_src, values_src, times_dst):
    times_src = np.asarray(times_src, dtype=float)
    values_src = np.asarray(values_src, dtype=bool)
    times_dst = np.asarray(times_dst, dtype=float)

    valid = np.isfinite(times_src)
    if valid.sum() < 1:
        return np.zeros_like(times_dst, dtype=bool)

    f = interp1d(
        times_src[valid],
        values_src[valid].astype(float),
        kind="nearest",
        bounds_error=False,
        fill_value=0.0,
        assume_sorted=True,
    )
    return np.asarray(f(times_dst) > 0.5, dtype=bool)


def _median_filter_pitch(f0: np.ndarray, width: int = MEDIAN_FILTER_WIDTH) -> np.ndarray:
    f0 = np.asarray(f0, dtype=float).copy()
    if len(f0) == 0:
        return f0

    out = f0.copy()
    half = width // 2

    for i in range(len(f0)):
        lo = max(0, i - half)
        hi = min(len(f0), i + half + 1)
        window = f0[lo:hi]
        window = window[np.isfinite(window) & (window > 0)]
        if window.size:
            out[i] = np.median(window)

    return out


def _prepare_common_grid(
    t_ref,
    f0_ref,
    v_ref,
    t_est,
    f0_est,
    v_est,
    fps: float = COMMON_GRID_FPS,
):
    t_ref = np.asarray(t_ref, dtype=float)
    f0_ref = np.asarray(f0_ref, dtype=float)
    v_ref = np.asarray(v_ref, dtype=bool)

    t_est = np.asarray(t_est, dtype=float)
    f0_est = np.asarray(f0_est, dtype=float)
    v_est = np.asarray(v_est, dtype=bool)

    if len(t_ref) == 0 or len(t_est) == 0:
        return None

    t0 = max(float(t_ref[0]), float(t_est[0]))
    t1 = min(float(t_ref[-1]), float(t_est[-1]))

    if not (t1 > t0 + 1e-6):
        return None

    n_points = max(MIN_COMMON_GRID_POINTS, int((t1 - t0) * fps) + 1)
    grid = np.linspace(t0, t1, n_points)

    f_ref = _interp_masked(t_ref, f0_ref, grid, kind="linear")
    f_est = _interp_masked(t_est, f0_est, grid, kind="linear")

    v_ref_grid = _interp_bool(t_ref, v_ref, grid)
    v_est_grid = _interp_bool(t_est, v_est, grid)

    f_ref[~v_ref_grid] = np.nan
    f_est[~v_est_grid] = np.nan

    f_ref = _median_filter_pitch(f_ref, width=MEDIAN_FILTER_WIDTH)
    f_est = _median_filter_pitch(f_est, width=MEDIAN_FILTER_WIDTH)

    return grid, f_ref, v_ref_grid, f_est, v_est_grid


def _dtw_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if len(X) == 0 or len(Y) == 0:
        return 0.0

    D = cdist(X, Y, metric="cosine")
    D = np.nan_to_num(D, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32)

    n, m = D.shape
    # Anti-diagonal vectorized DTW: O(n+m) Python iterations, all inner work in numpy
    acc = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    acc[0, 0] = 0.0
    for k in range(1, n + m + 1):
        i_lo = max(1, k - m)
        i_hi = min(n, k - 1) + 1
        ii = np.arange(i_lo, i_hi)
        jj = k - ii                         # j = k - i, ranges over valid (i,j) pairs
        diag = acc[ii - 1, jj - 1]
        up   = acc[ii - 1, jj]
        left = acc[ii,     jj - 1]
        acc[ii, jj] = D[ii - 1, jj - 1] + np.minimum(np.minimum(diag, up), left)

    cost = float(acc[n, m]) / max(n + m, 1)
    sim = 1.0 - cost
    return float(np.clip(sim, 0.0, 1.0))

# =========================================================
# MIR-EVAL MELODY METRICS
# =========================================================

def mir_melody_scores(t_ref, f0_ref, v_ref, t_est, f0_est, v_est) -> Dict[str, float]:
    prep = _prepare_common_grid(
        t_ref, f0_ref, v_ref,
        t_est, f0_est, v_est,
        fps=COMMON_GRID_FPS,
    )

    if prep is None:
        return {
            "overall_accuracy": 0.0,
            "raw_pitch_accuracy": 0.0,
            "raw_chroma_accuracy": 0.0,
            "voicing_recall": 0.0,
            "voicing_false_alarm": 0.0,
        }

    _, f_ref_hz, v_ref_grid, f_est_hz, v_est_grid = prep

    ref_cent = _hz_to_cents(f_ref_hz)
    est_cent = _hz_to_cents(f_est_hz)

    ref_cent_0 = np.where(v_ref_grid & np.isfinite(ref_cent), ref_cent, 0.0)
    est_cent_0 = np.where(v_est_grid & np.isfinite(est_cent), est_cent, 0.0)

    try:
        scores = {
            "overall_accuracy": mir_eval.melody.overall_accuracy(v_ref_grid, ref_cent_0, v_est_grid, est_cent_0),
            "raw_pitch_accuracy": mir_eval.melody.raw_pitch_accuracy(v_ref_grid, ref_cent_0, v_est_grid, est_cent_0),
            "raw_chroma_accuracy": mir_eval.melody.raw_chroma_accuracy(v_ref_grid, ref_cent_0, v_est_grid, est_cent_0),
            "voicing_recall": mir_eval.melody.voicing_recall(v_ref_grid, v_est_grid),
            "voicing_false_alarm": mir_eval.melody.voicing_false_alarm(v_ref_grid, v_est_grid),
        }
    except Exception:
        scores = {
            "overall_accuracy": 0.0,
            "raw_pitch_accuracy": 0.0,
            "raw_chroma_accuracy": 0.0,
            "voicing_recall": 0.0,
            "voicing_false_alarm": 0.0,
        }

    return {k: _safe_float(v, 0.0) for k, v in scores.items()}

# =========================================================
# CONTOUR
# =========================================================

CHROMA_HOP = 1600    # ~10 FPS at 16kHz — keeps DTW matrix small (60s → ~600 frames)

def _load_chroma(audio_path: str):
    """Load audio and compute chroma_cqt. Returns (12, T) array or None."""
    sr = SAMPLE_RATE
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
    except Exception:
        return None
    if len(y) == 0:
        return None
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=CHROMA_HOP, norm=2)


def chroma_contour_dtw_distance(audio_ref: str, audio_est: str) -> Dict[str, float]:
    """
    Chroma-based contour DTW similarity for instrumental music.

    Uses 12-dim chroma vectors instead of F0 to handle polyphony.
    Pitch-shift normalization via exhaustive 12-semitone circular shift search.
    Returns contour_dtw_similarity in [0, 1].
    """
    C_ref = _load_chroma(audio_ref)
    C_est = _load_chroma(audio_est)

    if C_ref is None or C_est is None:
        return {"contour_dtw_similarity": 0.0}

    n = min(C_ref.shape[1], C_est.shape[1])
    if n < MIN_CONTOUR_VALID_POINTS + 1:
        return {"contour_dtw_similarity": 0.0}

    C_ref = C_ref[:, :n]
    C_est = C_est[:, :n]

    # first-order difference along time axis — shape (n-1, 12)
    dC_ref = np.diff(C_ref, axis=1).T
    dC_est = np.diff(C_est, axis=1).T

    if len(dC_ref) < MIN_CONTOUR_DIFF_POINTS:
        return {"contour_dtw_similarity": 0.0}

    # pitch-shift normalization: try all 12 circular shifts, keep best
    best = 0.0
    for shift in range(12):
        shifted = np.roll(dC_est, shift, axis=1)
        sim = _dtw_cosine(dC_ref, shifted)
        if sim > best:
            best = sim

    return {"contour_dtw_similarity": _safe_float(best, 0.0)}


def chroma_motif_ngram_overlap(audio_ref: str, audio_est: str, n: int = MOTIF_NGRAM_N) -> Dict[str, float]:
    """
    Chroma-based motif n-gram overlap for instrumental music.

    Quantizes chroma first-order diff into 12-bin interval tokens, then computes
    n-gram jaccard and recall. Pitch-shift normalization via all 12 circular shifts
    (keeps the shift that maximises recall).
    """
    C_ref = _load_chroma(audio_ref)
    C_est = _load_chroma(audio_est)

    if C_ref is None or C_est is None:
        return {"jaccard": 0.0, "recall": 0.0}

    n_frames = min(C_ref.shape[1], C_est.shape[1])
    if n_frames < 2:
        return {"jaccard": 0.0, "recall": 0.0}

    C_ref = C_ref[:, :n_frames]
    C_est = C_est[:, :n_frames]

    # dominant pitch class per frame -> interval token sequence
    def _tokens(C: np.ndarray) -> List[int]:
        pitch_seq = np.argmax(C, axis=0).astype(int)   # (T,)
        intervals = np.diff(pitch_seq) % 12             # semitone intervals mod 12
        return intervals.tolist()

    def _grams(xs):
        return {tuple(xs[i:i + n]) for i in range(len(xs) - n + 1)} if len(xs) >= n else set()

    tokens_ref = _tokens(C_ref)
    grams_ref = _grams(tokens_ref)

    if not grams_ref:
        return {"jaccard": 0.0, "recall": 0.0}

    # try all 12 pitch shifts on est, keep best recall
    best_jaccard, best_recall = 0.0, 0.0
    for shift in range(12):
        C_shifted = np.roll(C_est, shift, axis=0)
        tokens_est = _tokens(C_shifted)
        grams_est = _grams(tokens_est)
        inter = len(grams_ref & grams_est)
        union = len(grams_ref | grams_est)
        recall = inter / (len(grams_ref) + EPSILON)
        jaccard = inter / (union + EPSILON)
        if recall > best_recall:
            best_recall = recall
            best_jaccard = jaccard

    return {
        "jaccard": _safe_float(best_jaccard, 0.0),
        "recall": _safe_float(best_recall, 0.0),
    }


def contour_dtw_distance(t_ref, f0_ref, t_est, f0_est) -> Dict[str, float]:
    t_ref = np.asarray(t_ref, dtype=float)
    f0_ref = np.asarray(f0_ref, dtype=float)
    t_est = np.asarray(t_est, dtype=float)
    f0_est = np.asarray(f0_est, dtype=float)

    if len(t_ref) < 2 or len(t_est) < 2:
        return {"contour_dtw_similarity": 0.0}

    t0 = max(float(t_ref[0]), float(t_est[0]))
    t1 = min(float(t_ref[-1]), float(t_est[-1]))

    if not (t1 > t0 + 1e-6):
        return {"contour_dtw_similarity": 0.0}

    grid = np.linspace(t0, t1, int((t1 - t0) * CONTOUR_GRID_FPS) + 1)

    f_ref = _interp_masked(t_ref, f0_ref, grid, kind="linear")
    f_est = _interp_masked(t_est, f0_est, grid, kind="linear")

    valid = np.isfinite(f_ref) & np.isfinite(f_est) & (f_ref > 0) & (f_est > 0)
    if valid.sum() < MIN_CONTOUR_VALID_POINTS:
        return {"contour_dtw_similarity": 0.0}

    ref_cents = _hz_to_cents(f_ref[valid])
    est_cents = _hz_to_cents(f_est[valid])

    d_ref = np.diff(ref_cents)
    d_est = np.diff(est_cents)

    if len(d_ref) < MIN_CONTOUR_DIFF_POINTS or len(d_est) < MIN_CONTOUR_DIFF_POINTS:
        return {"contour_dtw_similarity": 0.0}

    sim = _dtw_cosine(d_ref[:, None], d_est[:, None])
    return {"contour_dtw_similarity": _safe_float(sim, 0.0)}

# =========================================================
# MOTIFS
# =========================================================

def interval_tokens(
    t: np.ndarray,
    f0: np.ndarray,
    v: np.ndarray = None,
    fps: float = MOTIF_GRID_FPS,
    min_voiced_run: int = MIN_VOICED_RUN,
) -> List[int]:
    t = np.asarray(t, dtype=float)
    f0 = np.asarray(f0, dtype=float)

    if v is None:
        v = np.isfinite(f0) & (f0 > 0)
    else:
        v = np.asarray(v, dtype=bool)

    if len(t) < 2:
        return []

    grid = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * fps) + 1)
    f_grid = _interp_masked(t, f0, grid, kind="linear")
    v_grid = _interp_bool(t, v, grid)

    f_grid[~v_grid] = np.nan
    cents = _hz_to_cents(f_grid)

    valid = np.isfinite(cents)
    tokens: List[int] = []

    i = 0
    while i < len(cents):
        if not valid[i]:
            i += 1
            continue

        j = i
        while j < len(cents) and valid[j]:
            j += 1

        if (j - i) >= min_voiced_run:
            run = cents[i:j]
            steps = np.diff(run) / SEMITONE_IN_CENTS
            steps = np.round(steps).astype(int)
            tokens.extend(int(step) for step in steps if np.isfinite(step))

        i = j

    return tokens


def motif_ngram_overlap(
    t_ref,
    f0_ref,
    t_est,
    f0_est,
    v_ref=None,
    v_est=None,
    n: int = MOTIF_NGRAM_N,
) -> Dict[str, float]:
    tokens_ref = interval_tokens(t_ref, f0_ref, v_ref)
    tokens_est = interval_tokens(t_est, f0_est, v_est)

    def grams(xs):
        return {tuple(xs[i:i + n]) for i in range(len(xs) - n + 1)} if len(xs) >= n else set()

    grams_ref = grams(tokens_ref)
    grams_est = grams(tokens_est)

    if not grams_ref and not grams_est:
        return {"jaccard": 1.0, "recall": 1.0}

    if not grams_ref:
        return {"jaccard": 0.0, "recall": 0.0}

    inter = len(grams_ref & grams_est)
    union = len(grams_ref | grams_est)

    return {
        "jaccard": float(inter / (union + EPSILON)),
        "recall": float(inter / (len(grams_ref) + EPSILON)),
    }

# =========================================================
# MAIN
# =========================================================

def melody_score(audio_ref: str, audio_est: str, use_essentia: bool = False) -> Dict[str, Any]:
    t_ref, f0_ref, v_ref = f0_librosa(audio_ref)    # 
    t_est, f0_est, v_est = f0_librosa(audio_est)

    if len(t_ref) == 0 or len(t_est) == 0:
        return {"error": "F0 extraction failed"}

    out = {}
    out["mir_melody"] = mir_melody_scores(t_ref, f0_ref, v_ref, t_est, f0_est, v_est)
    out["contour_dtw"] = contour_dtw_distance(t_ref, f0_ref, t_est, f0_est)
    out["motif_overlap_n3"] = motif_ngram_overlap(
        t_ref, f0_ref, t_est, f0_est, v_ref, v_est, n=MOTIF_NGRAM_N
    )
    return out


def melody_score_pitch_shift_aware(audio_ref: str, audio_est: str, n_steps: int = 0) -> Dict[str, Any]:
    return {
        "normalized": {
            "contour_dtw": chroma_contour_dtw_distance(audio_ref, audio_est),
            "motif_overlap_n3": chroma_motif_ngram_overlap(audio_ref, audio_est, n=MOTIF_NGRAM_N),
        }
    }


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    from collections import defaultdict

    ORIG_DIR = Path(
        "/deepfreeze/sshfs/xuxin/musecpeval/objectivate_validation/"
        "music-context-preservation-benchmark/music/original_audio"
    )
    EDITS_ROOT = Path(
        "/deepfreeze/sshfs/xuxin/musecpeval/objectivate_validation/"
        "music-context-preservation-benchmark/music/edits_audio"
    )

    parser = argparse.ArgumentParser(description="Melody metrics for instrumental music")
    parser.add_argument(
        "edit_subdir",
        nargs="?",
        default=None,
        help=(
            "Edit subdirectory relative to edits_audio/, e.g. 'harmony/+7semitone'. "
            "Omit to run all leaf directories."
        ),
    )
    parser.add_argument("--orig-dir", default=str(ORIG_DIR), help="Original audio directory")
    parser.add_argument("--edits-root", default=str(EDITS_ROOT), help="Edits root directory")
    args = parser.parse_args()

    orig_dir = Path(args.orig_dir)
    edits_root = Path(args.edits_root)

    if args.edit_subdir:
        target = edits_root / args.edit_subdir
        if not target.is_dir():
            print(f"Directory not found: {target}", file=sys.stderr)
            sys.exit(1)
        edit_dirs = [target]
    else:
        edit_dirs = sorted(
            p for p in edits_root.rglob("*")
            if p.is_dir() and not any(c.is_dir() for c in p.iterdir())
        )

    if not edit_dirs:
        print("No edit directories found.", file=sys.stderr)
        sys.exit(1)

    def flatten_scores(scores: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in scores.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.update(flatten_scores(v, full_key))
            elif isinstance(v, (int, float)) and np.isfinite(float(v)):
                out[full_key] = float(v)
        return out

    for edit_dir in edit_dirs:
        pairs = sorted(
            (f, edit_dir / f.name)
            for f in orig_dir.glob("*.wav")
            if (edit_dir / f.name).exists()
        )
        if not pairs:
            continue

        category = edit_dir.relative_to(edits_root)
        print(f"\n=== {category} ({len(pairs)} pairs) ===", flush=True)

        accumulator: dict[str, list[float]] = defaultdict(list)
        flat_keys: list[str] = []

        for ref_path, est_path in pairs:
            print(f"  {ref_path.name} ...", end=" ", flush=True)
            try:
                scores = melody_score_pitch_shift_aware(str(ref_path), str(est_path))
            except Exception as e:
                print(f"ERROR: {e}", file=sys.stderr)
                continue

            flat = flatten_scores(scores)
            for k, v in flat.items():
                accumulator[k].append(v)
                if k not in flat_keys:
                    flat_keys.append(k)
            print("ok")

        if not flat_keys:
            continue

        print(f"\n{'Metric':<55}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
        print("-" * 80)
        for k in flat_keys:
            vals = accumulator[k]
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"{k:<55}  {mean:8.4f}  {std:8.4f}  {len(vals):4d}")