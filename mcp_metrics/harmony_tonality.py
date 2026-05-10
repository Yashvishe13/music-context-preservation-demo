# harmony_tonality.py

from typing import Dict, Any, Tuple, List
from functools import lru_cache
import numpy as np
import librosa
import mir_eval
from utils import circle_of_fifths_distance, cosine_sim, dtw_cosine, _safe_float

# =========================================================
# HYPERPARAMETERS
# =========================================================

SAMPLE_RATE = 22050
HOP_LENGTH = 512

CHROMA_METHOD = "cqt"          # "cqt", "stft", "cens"
CHROMA_NORM = np.inf
CHROMA_SMOOTH_FRAMES = 9

KEY_PROFILE_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=float,
)
KEY_PROFILE_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=float,
)

PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CHORD_TEMPLATE_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
CHORD_TEMPLATE_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

CHORD_FRAME_SMOOTH = 5
MIN_VALID_DURATION_SEC = 1e-6


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _safe_float(x, default=np.nan):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _load_audio(audio_path: str, sr: int = SAMPLE_RATE):
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    y = np.nan_to_num(y, nan=0.0)
    return y, sr


def _moving_average_1d(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(x, kernel, mode="same")


def _smooth_chroma(chroma: np.ndarray, win: int = CHROMA_SMOOTH_FRAMES) -> np.ndarray:
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        return chroma
    out = np.empty_like(chroma, dtype=float)
    for i in range(12):
        out[i] = _moving_average_1d(chroma[i], win)
    return out


def _framewise_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return X
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + eps)


@lru_cache(maxsize=128)
def _extract_chroma(audio_path: str, sr: int = SAMPLE_RATE, method: str = CHROMA_METHOD) -> np.ndarray:
    y, sr = _load_audio(audio_path, sr=sr)

    if method == "stft":
        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            norm=CHROMA_NORM,
        )
    elif method == "cens":
        chroma = librosa.feature.chroma_cens(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
        )
    else:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH,
            norm=CHROMA_NORM,
        )

    chroma = np.asarray(chroma, dtype=float)
    chroma = _smooth_chroma(chroma, win=CHROMA_SMOOTH_FRAMES)
    return chroma


def _best_transposition_cosine(mean_ref: np.ndarray, mean_est: np.ndarray) -> Tuple[float, int]:
    best_score = -1.0
    best_shift = 0

    for shift in range(12):
        rolled = np.roll(mean_est, shift)
        score = cosine_sim(mean_ref, rolled)
        if score > best_score:
            best_score = score
            best_shift = shift

    return float(best_score), int(best_shift)


def _best_transposition_dtw(C0: np.ndarray, C1: np.ndarray) -> Tuple[float, int]:
    best_score = -1.0
    best_shift = 0

    for shift in range(12):
        rolled = np.roll(C1, shift, axis=1)
        score = dtw_cosine(C0, rolled)
        if score > best_score:
            best_score = score
            best_shift = shift

    return float(best_score), int(best_shift)


def _major_minor_key_from_chroma(mean_chroma: np.ndarray) -> Dict[str, Any]:
    best_key = None
    best_scale = None
    best_corr = -np.inf

    mean_chroma = np.asarray(mean_chroma, dtype=float).reshape(-1)
    if mean_chroma.size != 12 or not np.isfinite(mean_chroma).any():
        return {"error": "invalid chroma"}

    for i in range(12):
        for profile, scale_name in (
            (KEY_PROFILE_MAJOR, "major"),
            (KEY_PROFILE_MINOR, "minor"),
        ):
            rotated = np.roll(profile, i)
            corr = np.corrcoef(mean_chroma, rotated)[0, 1]
            if np.isfinite(corr) and corr > best_corr:
                best_corr = corr
                best_key = PITCH_NAMES[i]
                best_scale = scale_name

    return {
        "key": best_key,
        "scale": best_scale,
        "strength": _safe_float(best_corr, 0.0),
    }


def _build_chord_templates() -> Dict[str, np.ndarray]:
    templates = {}
    for i, name in enumerate(PITCH_NAMES):
        templates[f"{name}:maj"] = np.roll(CHORD_TEMPLATE_MAJOR, i)
        templates[f"{name}:min"] = np.roll(CHORD_TEMPLATE_MINOR, i)
    return templates


def _smooth_labels(labels: List[str], win: int = CHORD_FRAME_SMOOTH) -> List[str]:
    if win <= 1 or len(labels) == 0:
        return labels

    half = win // 2
    out = []
    for i in range(len(labels)):
        lo = max(0, i - half)
        hi = min(len(labels), i + half + 1)
        chunk = labels[lo:hi]
        vals, counts = np.unique(chunk, return_counts=True)
        out.append(vals[np.argmax(counts)])
    return out


# =========================================================
# KEY
# =========================================================

def key_scale_music21(audio_path: str) -> Dict[str, Any]:
    try:
        chroma = _extract_chroma(audio_path, sr=SAMPLE_RATE, method=CHROMA_METHOD)
        mean_chroma = chroma.mean(axis=1)
        return _major_minor_key_from_chroma(mean_chroma)
    except Exception as exc:
        return {"error": f"key extraction failed: {exc}"}


def key_relatedness(ref_key: str, ref_scale: str, est_key: str, est_scale: str) -> Dict[str, Any]:
    steps, norm = circle_of_fifths_distance(ref_key, ref_scale, est_key, est_scale)
    if steps is None:
        return {"distance_steps": None, "distance_norm_0to1": None}
    return {
        "distance_steps": int(steps),
        "distance_norm_0to1": float(norm),
    }


# =========================================================
# CHROMA
# =========================================================

def chroma_similarity(
    audio_ref: str,
    audio_est: str,
    sr: int = SAMPLE_RATE,
    method: str = CHROMA_METHOD,
) -> Dict[str, float]:
    C0 = _extract_chroma(audio_ref, sr=sr, method=method).T   # [frames, 12]
    C1 = _extract_chroma(audio_est, sr=sr, method=method).T   # [frames, 12]

    if len(C0) == 0 or len(C1) == 0:
        return {
            "chroma_dtw_cosine": 0.0,
            "mean_chroma_cosine": 0.0,
            "best_shift_chroma_dtw_cosine": 0.0,
            "best_shift_mean_chroma_cosine": 0.0,
            "best_shift_steps": 0.0,
        }

    C0n = _framewise_normalize(C0)
    C1n = _framewise_normalize(C1)

    mean0 = C0n.mean(axis=0)
    mean1 = C1n.mean(axis=0)

    raw_dtw = dtw_cosine(C0n, C1n)
    raw_mean = cosine_sim(mean0, mean1)

    best_mean, best_mean_shift = _best_transposition_cosine(mean0, mean1)
    best_dtw, best_dtw_shift = _best_transposition_dtw(C0n, C1n)

    # usually these should agree, but keep the DTW-derived shift as the main one
    return {
        "chroma_dtw_cosine": _safe_float(raw_dtw, 0.0),
        "mean_chroma_cosine": _safe_float(raw_mean, 0.0),
        "best_shift_chroma_dtw_cosine": _safe_float(best_dtw, 0.0),
        "best_shift_mean_chroma_cosine": _safe_float(best_mean, 0.0),
        "best_shift_steps": float(best_dtw_shift),
    }


# =========================================================
# CHORDS
# =========================================================

def chord_sequences_librosa(audio_path: str):
    try:
        chroma = _extract_chroma(audio_path, sr=SAMPLE_RATE, method=CHROMA_METHOD)
        templates = _build_chord_templates()

        frame_duration = HOP_LENGTH / SAMPLE_RATE
        labels = []

        for frame in chroma.T:
            best_label = "N"
            best_score = -np.inf

            for chord_name, template in templates.items():
                score = float(np.dot(frame, template))
                if score > best_score:
                    best_score = score
                    best_label = chord_name

            labels.append(best_label)

        labels = _smooth_labels(labels, win=CHORD_FRAME_SMOOTH)

        if len(labels) == 0:
            return None, None

        intervals = []
        out_labels = []

        start = 0.0
        current = labels[0]

        for i, lab in enumerate(labels[1:], start=1):
            if lab != current:
                end = i * frame_duration
                if end - start > MIN_VALID_DURATION_SEC:
                    intervals.append([start, end])
                    out_labels.append(current)
                start = end
                current = lab

        final_end = len(labels) * frame_duration
        if final_end - start > MIN_VALID_DURATION_SEC:
            intervals.append([start, final_end])
            out_labels.append(current)

        if len(intervals) == 0:
            return None, None

        return np.asarray(intervals, dtype=float), out_labels

    except Exception:
        return None, None


def chord_similarity_mireval(audio_ref: str, audio_est: str) -> Dict[str, float]:
    try:
        iv_ref, lb_ref = chord_sequences_librosa(audio_ref)
        iv_est, lb_est = chord_sequences_librosa(audio_est)

        if iv_ref is None or iv_est is None or len(iv_ref) == 0 or len(iv_est) == 0:
            return {"error": "chord extraction failed"}

        scores = mir_eval.chord.evaluate(iv_ref, lb_ref, iv_est, lb_est)
        return {f"mir_{k}": float(v) for k, v in scores.items()}

    except Exception as exc:
        return {"error": f"chord evaluation failed: {exc}"}


# =========================================================
# MAIN
# =========================================================

def harmony_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    result = {}

    key_ref = key_scale_music21(audio_ref)
    key_est = key_scale_music21(audio_est)

    result["key_ref"] = key_ref
    result["key_est"] = key_est

    if "key" in key_ref and "key" in key_est:
        result["key_relatedness"] = key_relatedness(
            key_ref["key"], key_ref["scale"],
            key_est["key"], key_est["scale"],
        )

    result["chroma_similarity"] = chroma_similarity(audio_ref, audio_est)
    result["chord_similarity"] = chord_similarity_mireval(audio_ref, audio_est)

    return result