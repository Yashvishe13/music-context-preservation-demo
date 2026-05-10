from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import mir_eval

# ---------- Basic Tools ----------
def _bounds_to_intervals(bounds):
    """1D boundaries -> (n,2) intervals; if already (n,2), return float view directly."""
    b = np.asarray(bounds, float)
    if b.ndim == 1:
        # Remove duplicates, sort, remove non-finite values
        b = b[np.isfinite(b)]
        b = np.unique(b)
        if b.size < 2:
            return np.empty((0, 2), dtype=float)
        return np.column_stack([b[:-1], b[1:]])
    # Already (n,2)
    return b.astype(float, copy=False)

def _shift_intervals(intv: np.ndarray, d: float):
    """Shift intervals globally; maintain shape."""
    intv = np.asarray(intv, float)
    if intv.size == 0:
        return intv
    return intv + np.array([d, d], dtype=float)

def _crop_overlap_for_detection(ref_int: np.ndarray, est_int: np.ndarray):
    """Crop to [t0, t1] common overlap window (and t0 ≥ 0), remove intervals with length ≤ 0.
    Return (ref_c, est_c); if no overlap, return (None, None).
    """
    ref_int = np.asarray(ref_int, float)
    est_int = np.asarray(est_int, float)
    if ref_int.size == 0 or est_int.size == 0:
        return None, None

    # Common time window: start point takes max of both and 0; end point takes min of both
    t0 = max(0.0, float(ref_int[0, 0]), float(est_int[0, 0]))
    t1 = min(float(ref_int[-1, 1]), float(est_int[-1, 1]))
    if not (t1 > t0):
        return None, None

    def _clip_valid(intv):
        if intv.size == 0:
            return None
        S = np.maximum(intv[:, 0], t0)
        E = np.minimum(intv[:, 1], t1)
        keep = (E - S) > 1e-10
        if not np.any(keep):
            return None
        return np.column_stack([S[keep], E[keep]])

    ref_c = _clip_valid(ref_int)
    est_c = _clip_valid(est_int)
    if ref_c is None or est_c is None:
        return None, None
    return ref_c, est_c

# ---------- Search for Best Shift ----------
def best_boundary_shift(ref_bounds, est_bounds, window=0.5, search_range=2.0, step=0.02):
    """Search for global shift Δ in [-search_range, +search_range] to maximize boundary F."""
    ref_int = _bounds_to_intervals(ref_bounds)
    est_int = _bounds_to_intervals(est_bounds)
    best = {"F": -1.0, "delta": 0.0, "P": 0.0, "R": 0.0}

    if ref_int.size == 0 or est_int.size == 0:
        return best

    for d in np.arange(-search_range, search_range + 1e-12, step):
        est_shift = _shift_intervals(est_int, d)
        ref_c, est_c = _crop_overlap_for_detection(ref_int, est_shift)
        if ref_c is None or est_c is None:
            continue
        P, R, F = mir_eval.segment.detection(ref_c, est_c, window=window, trim=True)
        if F > best["F"]:
            best.update({"F": float(F), "delta": float(d), "P": float(P), "R": float(R)})
    return best

def boundary_report_multi(ref_bounds, est_bounds, windows=(0.5, 1.0, 2.0)):
    """Multi-tolerance window boundary metrics; first convert to intervals and crop to common overlap window."""
    ref_int = _bounds_to_intervals(ref_bounds)
    est_int = _bounds_to_intervals(est_bounds)
    out = {}
    ref_c, est_c = _crop_overlap_for_detection(ref_int, est_int)
    if ref_c is None or est_c is None:
        for w in windows:
            out[f"boundary_P@{w}s"] = 0.0
            out[f"boundary_R@{w}s"] = 0.0
            out[f"boundary_F@{w}s"] = 0.0
        return out

    for w in windows:
        P, R, F = mir_eval.segment.detection(ref_c, est_c, window=w, trim=True)
        out[f"boundary_P@{w}s"] = float(P)
        out[f"boundary_R@{w}s"] = float(R)
        out[f"boundary_F@{w}s"] = float(F)
    return out

# ---------- Chroma-based Segment Labels ----------
def _chroma_segment_labels(audio_path: str, bounds: np.ndarray) -> List[str]:
    """
    Assign repeatable segment labels based on dominant chroma pitch class.
    Unlike sequential labels (S0, S1, ...), these can repeat (e.g. A-B-A),
    making ARI meaningful.
    """
    _PITCH_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop = 512
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        labels = []
        for i in range(len(bounds) - 1):
            f0 = max(0, int(bounds[i] * sr / hop))
            f1 = min(chroma.shape[1], int(bounds[i + 1] * sr / hop))
            if f1 > f0:
                labels.append(_PITCH_NAMES[int(np.argmax(chroma[:, f0:f1].mean(axis=1)))])
            else:
                labels.append("C")
        return labels if labels else ["C"]
    except Exception:
        return [f"S{i}" for i in range(max(0, len(bounds) - 1))]


# ---------- MSAF ----------
def msaf_segments(audio_path: str, feature: str = "pcp", algo: str = "sf"):
    try:
        import scipy
        if not hasattr(scipy, 'inf'):
            scipy.inf = np.inf
        import msaf

        # Duration (for adding tail boundary)
        duration = None
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = float(info.frames) / float(info.samplerate)
        except Exception:
            try:
                import librosa
                y, sr = librosa.load(audio_path, sr=None, mono=True)
                duration = float(len(y)) / float(sr)
            except Exception:
                pass

        bounds, labels = None, None
        try:
            try:
                est = msaf.run.process(audio_path, boundaries_id=algo, labels_id="scluster")
            except Exception:
                est = msaf.run.process(audio_path, boundaries_id=algo, labels_id=None)
            bounds = np.asarray(est["est_times"], float)
            labels = est.get("est_labels", None)
        except Exception:
            try:
                bounds, labels = msaf.process(audio_path)
                bounds = np.asarray(bounds, float)
                labels = None  # will be replaced by chroma labels below
            except Exception:
                return None, None

        # Normalize: ≥0, sort and remove duplicates
        bounds = np.asarray(bounds, float)
        bounds = bounds[np.isfinite(bounds)]
        bounds = np.clip(bounds, 0.0, np.inf)
        bounds = np.unique(bounds)
        if bounds.size == 0 or bounds[0] > 1e-6:
            bounds = np.r_[0.0, bounds]
        if duration is not None and abs(bounds[-1] - duration) > 1e-3:
            if bounds[-1] < duration:
                bounds = np.r_[bounds, duration]
            else:
                bounds[-1] = duration

        # Align labels to number of intervals
        if labels is None:
            labels = _chroma_segment_labels(audio_path, bounds)
        else:
            labels = list(labels)
            need = max(0, len(bounds) - 1)
            if len(labels) < need:
                labels = labels + [labels[-1] if labels else "S"] * (need - len(labels))
            elif len(labels) > need:
                labels = labels[:need]

        return bounds, labels
    except Exception:
        return None, None

# ---------- Conversion and Cropping ----------
def bounds_to_intervals(bounds: np.ndarray) -> np.ndarray:
    return _bounds_to_intervals(bounds)

def crop_to_overlap(
    intervals: np.ndarray,
    labels: Optional[List],
    t_min: float,
    t_max: float
) -> Tuple[np.ndarray, Optional[List]]:
    if intervals.size == 0:
        return intervals, [] if labels is not None else None

    S = np.maximum(intervals[:, 0], t_min)
    E = np.minimum(intervals[:, 1], t_max)
    keep = (E - S) > 1e-10
    intervals_c = np.column_stack([S[keep], E[keep]])

    labels_c = None
    if labels is not None:
        labels_arr = np.asarray(labels, dtype=object)
        labels_c = labels_arr[keep].tolist()
        if len(labels_c) < len(intervals_c):
            labels_c += [labels_c[-1] if labels_c else "S"] * (len(intervals_c) - len(labels_c))
        elif len(labels_c) > len(intervals_c):
            labels_c = labels_c[:len(intervals_c)]
    return intervals_c, labels_c

# ---------- Evaluation ----------
def mir_segment_metrics(ref_bounds, est_bounds, ref_labels=None, est_labels=None) -> Dict[str, float]:
    ref_intervals = bounds_to_intervals(ref_bounds)
    est_intervals = bounds_to_intervals(est_bounds)

    if ref_intervals.size == 0 or est_intervals.size == 0:
        return {"boundary_p": 0.0, "boundary_r": 0.0, "boundary_f": 0.0}

    t0 = max(float(ref_intervals[0, 0]), float(est_intervals[0, 0]), 0.0)
    t1 = min(float(ref_intervals[-1, 1]), float(est_intervals[-1, 1]))
    if not (t1 > t0):
        return {"boundary_p": 0.0, "boundary_r": 0.0, "boundary_f": 0.0, "note": "no temporal overlap"}

    ref_intervals_c, ref_labels_c = crop_to_overlap(ref_intervals, ref_labels, t0, t1)
    est_intervals_c, est_labels_c = crop_to_overlap(est_intervals, est_labels, t0, t1)

    P, R, F = mir_eval.segment.detection(ref_intervals_c, est_intervals_c, window=0.5, trim=True)
    out = {"boundary_p": float(P), "boundary_r": float(R), "boundary_f": float(F)}

    if (ref_labels_c is not None and est_labels_c is not None and
        len(ref_labels_c) == len(ref_intervals_c) and len(est_labels_c) == len(est_intervals_c)):
        try:
            p, r, f = mir_eval.segment.pairwise(ref_intervals_c, ref_labels_c, est_intervals_c, est_labels_c)
            out.update({"pairwise_p": float(p), "pairwise_r": float(r), "pairwise_f": float(f)})
        except Exception as e:
            out["pairwise_error"] = f"{type(e).__name__}: {e}"
        try:
            ari = mir_eval.segment.ari(ref_intervals_c, ref_labels_c, est_intervals_c, est_labels_c)
            out["ari"] = float(ari)
        except Exception as e:
            out["ari"] = float("nan")
            out["ari_error"] = f"{type(e).__name__}: {e}"

    return out

# ---------- Entry Point ----------
def librosa_segments_fallback(audio_path: str):
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = float(len(y)) / float(sr)
        n_segments = max(2, min(12, int(round(duration / 10.0))))
        bounds = np.linspace(0.0, duration, n_segments + 1)
        labels = _chroma_segment_labels(audio_path, bounds)
        return bounds, labels
    except Exception:
        return np.array([0.0, 30.0]), ["S0"]

def structural_score(audio_ref: str, audio_est: str) -> Dict[str, Any]:
    rb, rl = msaf_segments(audio_ref)
    eb, el = msaf_segments(audio_est)
    if rb is None or eb is None:
        rb, rl = librosa_segments_fallback(audio_ref)
        eb, el = librosa_segments_fallback(audio_est)

    best = best_boundary_shift(rb, eb, window=0.5, search_range=2.0, step=0.02)
    eb_shifted = np.asarray(eb, float) + best["delta"]

    scores = mir_segment_metrics(rb, eb_shifted, rl, el)

    boundary_f_after_shift = max(0.0, best["F"])  # F might be -1, clamp to 0
    ari = scores.get("ari", 0.0)
    pairwise_f = scores.get("pairwise_f", 0.0)
    scores.update({
        "best_boundary_shift_sec": best["delta"],
        "boundary_f_after_shift": boundary_f_after_shift,
        "pairwise_f": float(pairwise_f),
        "structural_form_score": 0.7 * boundary_f_after_shift + 0.3 * ari
    })

    # raw_multi  = boundary_report_multi(rb, eb,         windows=(0.5, 1.0, 2.0))
    # shft_multi = boundary_report_multi(rb, eb_shifted, windows=(0.5, 1.0, 2.0))
    # scores.update({f"raw_{k}": v for k, v in raw_multi.items()})
    # scores.update({f"aligned_{k}": v for k, v in shft_multi.items()})

    return scores

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument("edit_dir", type=str, help="Path to editing subfolder (e.g. edits_audio/structural/AAA)")
    args = parser.parse_args()

    orig_dir = Path(
        "/deepfreeze/sshfs/xuxin/musecpeval/objectivate_validation/"
        "music-context-preservation-benchmark/music/original_audio"
    )
    edit_dir = Path(args.edit_dir)

    if not edit_dir.exists():
        print(f"Edit dir not found: {edit_dir}", file=sys.stderr)
        sys.exit(1)

    pairs = sorted(
        (f, edit_dir / f.name)
        for f in orig_dir.glob("*.wav")
        if (edit_dir / f.name).exists()
    )

    if not pairs:
        print("No matching pairs found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} pairs in {edit_dir}", flush=True)

    FLAT_KEYS = [
        "boundary_p", "boundary_r", "boundary_f",
        "boundary_f_after_shift",
        "pairwise_p", "pairwise_r", "pairwise_f",
        "ari",
        "structural_form_score",
        "best_boundary_shift_sec",
    ]

    accumulator: dict[str, list[float]] = defaultdict(list)

    for ref_path, est_path in pairs:
        print(f"  {ref_path.name} ...", end=" ", flush=True)
        try:
            scores = structural_score(str(ref_path), str(est_path))
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            continue

        for k in FLAT_KEYS:
            v = scores.get(k)
            if v is not None and np.isfinite(float(v)):
                accumulator[k].append(float(v))
        print("ok")

    print(f"\n--- Results: {edit_dir.name} ---")
    print(f"{'Metric':<35}  {'Mean':>8}  {'Std':>8}  {'N':>4}")
    print("-" * 60)
    for k in FLAT_KEYS:
        vals = accumulator.get(k, [])
        if not vals:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        print(f"{k:<35}  {mean:8.4f}  {std:8.4f}  {len(vals):4d}")