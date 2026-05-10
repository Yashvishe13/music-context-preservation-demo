from typing import Dict, Any, Tuple
import numpy as np
import librosa
import mir_eval
from utils import _safe_float

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 22050
BEAT_TRACK_UNITS = "time"
BEAT_TRACK_TRIM = True

DEFAULT_BEAT_PERIOD_SEC = 0.5
MIN_BEATS_FOR_PERIOD = 2

OFFSET_GOOD_FRACTION_OF_PERIOD = 0.6
OFFSET_GRID_HALF_WIDTH_IN_PERIODS = 0.5
OFFSET_GRID_STEPS = 41

BEAT_MATCH_TOLERANCE_SEC = 0.07
HALF_DOUBLE_RATIO = 2.0
EPSILON = 1e-8

INFO_GAIN_BINS = 41
INFO_GAIN_MAX_BITS = float(np.log2(INFO_GAIN_BINS))

# =========================================================
# INTERNAL HELPERS
# =========================================================


def _as_1d_float_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


def _normalize_beats(beats) -> np.ndarray:
    beats = _as_1d_float_array(beats)
    if beats.size == 0:
        return beats
    beats = np.unique(beats)
    return beats


def _folded_bpm_delta(bpm_ref: float, bpm_est: float) -> float:
    bpm_ref = _safe_float(bpm_ref, default=np.nan)
    bpm_est = _safe_float(bpm_est, default=np.nan)

    if not np.isfinite(bpm_ref) or not np.isfinite(bpm_est) or bpm_ref <= 0 or bpm_est <= 0:
        return np.nan

    candidates = [
        bpm_est,
        bpm_est * HALF_DOUBLE_RATIO,
        bpm_est / HALF_DOUBLE_RATIO,
    ]
    return float(min(abs(bpm_ref - c) for c in candidates if c > 0))


# =========================================================
# BEAT / TEMPO EXTRACTION
# =========================================================

def tempo_and_beats_librosa(audio_path: str, sr: int = SAMPLE_RATE) -> Tuple[float, np.ndarray]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y = np.nan_to_num(y, nan=0.0)

    tempo, beats = librosa.beat.beat_track(
        y=y,
        sr=sr,
        units=BEAT_TRACK_UNITS,
        trim=BEAT_TRACK_TRIM,
    )

    if hasattr(tempo, "__len__") and len(np.atleast_1d(tempo)) > 0:
        tempo_val = float(np.atleast_1d(tempo)[0])
    else:
        tempo_val = float(tempo)

    beats = _normalize_beats(beats)
    return tempo_val, beats


# =========================================================
# MIR_EVAL BEAT METRICS
# =========================================================

def beat_fmeasure_mireval(ref_beats, est_beats) -> Dict[str, float]:
    ref_beats = _normalize_beats(ref_beats)
    est_beats = _normalize_beats(est_beats)

    if ref_beats.size == 0 or est_beats.size == 0:
        return {
            "F-measure": 0.0,
            "Cemgil": 0.0,
            "Cemgil Best Metric Level": 0.0,
            "Goto": 0.0,
            "P-score": 0.0,
            "Correct Metric Level Continuous": 0.0,
            "Correct Metric Level Total": 0.0,
            "Any Metric Level Continuous": 0.0,
            "Any Metric Level Total": 0.0,
            "Information gain": 0.0,
        }

    try:
        scores = mir_eval.beat.evaluate(ref_beats, est_beats)
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        print(f"mir_eval error: {e}")
        return {
            "F-measure": 0.0,
            "Cemgil": 0.0,
            "Cemgil Best Metric Level": 0.0,
            "Goto": 0.0,
            "P-score": 0.0,
            "Correct Metric Level Continuous": 0.0,
            "Correct Metric Level Total": 0.0,
            "Any Metric Level Continuous": 0.0,
            "Any Metric Level Total": 0.0,
            "Information gain": 0.0,
        }


# =========================================================
# BEAT PERIOD / OFFSET
# =========================================================

def _estimate_beat_period_from_all_beats(all_beats_times: np.ndarray) -> float:
    all_beats_times = _normalize_beats(all_beats_times)

    if all_beats_times.size < MIN_BEATS_FOR_PERIOD:
        return DEFAULT_BEAT_PERIOD_SEC

    d = np.diff(all_beats_times)
    d = d[np.isfinite(d) & (d > 0)]

    if d.size == 0:
        return DEFAULT_BEAT_PERIOD_SEC

    return float(np.median(d))


def _best_offset_grid(
    ref_times: np.ndarray,
    est_times: np.ndarray,
    beat_period: float,
    tol: float,
) -> Tuple[float, int]:
    ref_times = _normalize_beats(ref_times)
    est_times = _normalize_beats(est_times)

    if ref_times.size == 0 or est_times.size == 0:
        return 0.0, 0

    diffs = []
    for t in ref_times:
        j = int(np.argmin(np.abs(est_times - t)))
        diffs.append(est_times[j] - t)

    diffs = np.asarray(diffs, dtype=float)

    good = np.abs(diffs) <= OFFSET_GOOD_FRACTION_OF_PERIOD * beat_period
    delta0 = float(np.median(diffs[good])) if np.any(good) else 0.0

    grid = np.linspace(
        delta0 - OFFSET_GRID_HALF_WIDTH_IN_PERIODS * beat_period,
        delta0 + OFFSET_GRID_HALF_WIDTH_IN_PERIODS * beat_period,
        OFFSET_GRID_STEPS,
    )

    def match_cnt(delta: float) -> int:
        i = 0
        j = 0
        tp = 0

        est_shift = est_times + delta

        while i < len(ref_times) and j < len(est_shift):
            d = est_shift[j] - ref_times[i]

            if abs(d) <= tol:
                tp += 1
                i += 1
                j += 1
            elif d < -tol:
                j += 1
            else:
                i += 1

        return tp

    counts = np.array([match_cnt(delta) for delta in grid], dtype=int)
    best_idx = int(np.argmax(counts))

    return float(grid[best_idx]), int(counts[best_idx])


# =========================================================
# MAIN API
# =========================================================

def rhythm_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    tempo_ref_bpm, beats_ref = tempo_and_beats_librosa(audio_ref)   # estimated tempo, downbeat time position list
    tempo_est_bpm, beats_est = tempo_and_beats_librosa(audio_est)

    beat_period_ref = _estimate_beat_period_from_all_beats(beats_ref)   # time between two beats
    beat_period_est = _estimate_beat_period_from_all_beats(beats_est)
    beat_period = float(np.nanmedian([beat_period_ref, beat_period_est]))

    best_offset_sec, best_offset_matches = _best_offset_grid(
        beats_ref,
        beats_est,
        beat_period=beat_period,
        tol=BEAT_MATCH_TOLERANCE_SEC,
    )

    beat_scores = beat_fmeasure_mireval(beats_ref, beats_est)   # F-measure, information gain

    out["tempo_ref_bpm"] = _safe_float(tempo_ref_bpm, 0.0)
    out["tempo_est_bpm"] = _safe_float(tempo_est_bpm, 0.0)

    # Raw BPM delta - beat difference
    raw_delta = abs(out["tempo_ref_bpm"] - out["tempo_est_bpm"])
    out["delta_bpm"] = float(raw_delta)

    # Folded BPM delta (robust to half/double tempo) - use this one
    out["delta_bpm_folded"] = _safe_float(
        _folded_bpm_delta(out["tempo_ref_bpm"], out["tempo_est_bpm"]),
        0.0,
    )

    out["beat_period_sec"] = _safe_float(beat_period, DEFAULT_BEAT_PERIOD_SEC)
    out["best_offset_sec"] = _safe_float(best_offset_sec, 0.0)
    out["best_offset_matches"] = int(best_offset_matches)

    out["n_ref_beats"] = int(len(beats_ref))
    out["n_est_beats"] = int(len(beats_est))

    out["beat_mir_eval"] = beat_scores
    return out


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    from collections import defaultdict

    orig_dir = Path(
        "/deepfreeze/sshfs/xuxin/musecpeval/objectivate_validation/"
        "music-context-preservation-benchmark/music/original_audio"
    )
    edit_dir = Path(
        "/deepfreeze/sshfs/xuxin/musecpeval/objectivate_validation/"
        "music-context-preservation-benchmark/music/edits_audio/rhythm/tempo+50"
    )

    pairs = sorted(
        (f, edit_dir / f.name)
        for f in orig_dir.glob("*.wav")
        if (edit_dir / f.name).exists()
    )

    if not pairs:
        print("No matching pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} pairs", flush=True)

    flat_keys: list[str] = []
    accumulator: dict[str, list[float]] = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = rhythm_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        for k, v in scores.items():
            if k == "beat_mir_eval":
                for sub_k, sub_v in v.items():
                    key = f"beat_mir_eval.{sub_k}"
                    accumulator[key].append(float(sub_v))
                    if key not in flat_keys:
                        flat_keys.append(key)
            elif isinstance(v, (int, float)):
                accumulator[k].append(float(v))
                if k not in flat_keys:
                    flat_keys.append(k)
        print("ok")

    print("\n--- Results ---")
    print(f"{'Metric':<45}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 70)
    for k in flat_keys:
        vals = accumulator[k]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"{k:<45}  {mean:8.4f}  {std:8.4f}  {len(vals):4d}")