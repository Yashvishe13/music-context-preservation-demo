import numpy as np

# fix deprecated numpy aliases
np.int = int
np.float = float
np.complex = complex
np.bool = bool
np.object = object
np.str = str

print("✓ numpy aliases patched")


import os
import musdb
import soundfile as sf
from tqdm import tqdm

# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = "musdb"
OUTPUT_DIR = "musdb_wav"
SUBSET = None          # None = all tracks, "train" = train only, "test" = test only

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LOAD DATASET (skip if WAVs already exist)
# =========================================================
existing_wavs = sorted(
    os.path.join(OUTPUT_DIR, f)
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".wav")
)

if existing_wavs:
    downloaded = existing_wavs
    print(f"✓ Found {len(downloaded)} existing WAV files in {OUTPUT_DIR}, skipping download.")
else:
    if SUBSET is None:
        db = musdb.DB(root=ROOT_DIR, download=True)
    else:
        db = musdb.DB(root=ROOT_DIR, subsets=SUBSET, download=True)

    tracks = db.tracks
    print(f"Total tracks found: {len(tracks)}")

    downloaded = []

    for track in tqdm(tracks, desc="Saving MUSDB tracks"):
        try:
            audio = track.audio
            sr = track.rate

            subset_name = getattr(track, "subset", "all")
            out_path = os.path.join(OUTPUT_DIR, f"{subset_name}_{track.name}.wav")

            sf.write(out_path, audio, sr)
            downloaded.append(out_path)

        except Exception as e:
            print(f"⚠️ Failed: {track.name} ({e})")

    print(f"✓ Saved {len(downloaded)} tracks to {OUTPUT_DIR}")
    print("Done.")


# =========================================================
# PITCH-SHIFT EDIT + EVAL PIPELINE
# For already-downloaded MUSDB WAV files in `downloaded`
# =========================================================
#
# Assumes these are already defined/imported in your Colab:
#   - downloaded  (list of original wav paths)
#   - harmony_score
#   - rhythm_score
#   - structural_score
#   - melody_score_pitch_shift_aware
#
# Example imports:
from harmony_tonality import harmony_score
from rhythm_meter import rhythm_score
from structural_form import structural_score
from melody_motifs import melody_score_pitch_shift_aware

# =========================================================

import os
import json
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# HYPERPARAMETERS
# =========================================================

PITCH_SHIFT_STEPS = 6
AUDIO_EDIT_DIR = "audio_edited/pitch_shift"
RESULTS_JSON_PATH = "pitch_shift_results.json"

# I/O parallelism for generating edited audio
NUM_AUDIO_WORKERS = 8

# Eval batching / checkpointing
EVAL_BATCH_SIZE = 32
SAVE_EVERY_BATCH = True

# =========================================================
# SETUP
# =========================================================

os.makedirs(AUDIO_EDIT_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================

def safe_float(x, default=np.nan):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def valid_vals(xs):
    out = []
    for x in xs:
        v = safe_float(x)
        if np.isfinite(v):
            out.append(v)
    return out


def summarize(vals):
    vals = valid_vals(vals)
    if not vals:
        return np.nan, np.nan, 0
    return float(np.median(vals)), float(np.std(vals)), len(vals)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_pitch_shifted_path(src_path: str, out_dir: str = AUDIO_EDIT_DIR) -> str:
    base = os.path.basename(src_path)
    return os.path.join(out_dir, base)


# =========================================================
# 1) CREATE PITCH-SHIFTED COPIES
# =========================================================

def create_pitch_shifted_file(src_path: str, n_steps: int = PITCH_SHIFT_STEPS) -> str:
    out_path = make_pitch_shifted_path(src_path)

    if os.path.exists(out_path):
        return out_path

    y, sr = librosa.load(src_path, sr=None, mono=True)
    y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    sf.write(out_path, y_shifted, sr)
    return out_path


expected_shifted = [make_pitch_shifted_path(p) for p in downloaded]
all_shifted_exist = all(os.path.exists(p) for p in expected_shifted)

if all_shifted_exist:
    pitch_shifted = expected_shifted
    print(f"✓ All {len(pitch_shifted)} pitch-shifted files already exist, skipping pitch-shift step.")
else:
    pitch_shifted = [None] * len(downloaded)

    with ThreadPoolExecutor(max_workers=NUM_AUDIO_WORKERS) as executor:
        future_to_idx = {
            executor.submit(create_pitch_shifted_file, src_path, PITCH_SHIFT_STEPS): idx
            for idx, src_path in enumerate(downloaded)
        }

        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Creating pitch-shifted audio"):
            idx = future_to_idx[future]
            try:
                pitch_shifted[idx] = future.result()
            except Exception as e:
                print(f"⚠️ Failed pitch shift for {downloaded[idx]}: {e}")
                pitch_shifted[idx] = None

pairs = [(ref, est) for ref, est in zip(downloaded, pitch_shifted) if est is not None]

n_shifted_on_disk = len([f for f in os.listdir(AUDIO_EDIT_DIR) if f.endswith(".wav")])
print(f"✓ Pitch-shifted files on disk: {n_shifted_on_disk}")
print(f"✓ Valid pitch-shifted pairs for eval: {len(pairs)}")

# =========================================================
# 2) EVALUATE
# =========================================================

def evaluate_pair(ref: str, est: str, n_steps: int = PITCH_SHIFT_STEPS) -> dict:
    return {
        "ref": ref,
        "est": est,
        "edit_type": "pitch_shift",
        "n_steps": n_steps,
        "harmony_tonality": harmony_score(ref, est),
        "rhythm_meter": rhythm_score(ref, est),
        "structural_form": structural_score(ref, est),
        "melodic_content": melody_score_pitch_shift_aware(ref, est, n_steps=n_steps),
    }


all_results = []

for batch_start in tqdm(range(0, len(pairs), EVAL_BATCH_SIZE), desc="Evaluating batches"):
    batch = pairs[batch_start:batch_start + EVAL_BATCH_SIZE]

    batch_results = []
    for ref, est in batch:
        try:
            batch_results.append(evaluate_pair(ref, est, PITCH_SHIFT_STEPS))
        except Exception as e:
            batch_results.append({
                "ref": ref,
                "est": est,
                "edit_type": "pitch_shift",
                "n_steps": PITCH_SHIFT_STEPS,
                "error": str(e),
            })

    all_results.extend(batch_results)

    if SAVE_EVERY_BATCH:
        save_json(RESULTS_JSON_PATH, all_results)

    print(f"✓ batch {batch_start // EVAL_BATCH_SIZE + 1} done ({min(batch_start + EVAL_BATCH_SIZE, len(pairs))}/{len(pairs)})")

save_json(RESULTS_JSON_PATH, all_results)
print(f"✓ Saved raw results to {RESULTS_JSON_PATH}")

# =========================================================
# 3) EXTRACT METRICS
# =========================================================

def extract_metrics(results):
    metrics = {
        # harmony
        "harmony_key_distance": [],
        "harmony_chroma_dtw": [],
        "harmony_chroma_mean": [],
        "harmony_best_shift_chroma_dtw": [],
        "harmony_best_shift_chroma_mean": [],
        "harmony_chord_root": [],
        "harmony_chord_majmin": [],

        # rhythm
        "rhythm_delta_bpm": [],
        "rhythm_delta_bpm_folded": [],
        "rhythm_beat_fmeasure": [],
        "rhythm_info_gain": [],

        # structure
        "structure_boundary_f": [],
        "structure_ari": [],
        "structure_score": [],

        # melody raw
        "melody_raw_overall_acc": [],
        "melody_raw_voicing_recall": [],
        "melody_raw_pitch_acc": [],
        "melody_raw_chroma_acc": [],
        "melody_raw_contour_dtw": [],
        "melody_raw_motif_jaccard": [],
        "melody_raw_motif_recall": [],

        # melody normalized
        "melody_norm_overall_acc": [],
        "melody_norm_voicing_recall": [],
        "melody_norm_pitch_acc": [],
        "melody_norm_chroma_acc": [],
        "melody_norm_contour_dtw": [],
        "melody_norm_motif_jaccard": [],
        "melody_norm_motif_recall": [],
    }

    for r in results:
        if "error" in r:
            continue

        h = r.get("harmony_tonality", {})
        rm = r.get("rhythm_meter", {})
        s = r.get("structural_form", {})
        m = r.get("melodic_content", {})

        # ---------------- harmony ----------------
        metrics["harmony_key_distance"].append(
            safe_float(h.get("key_relatedness", {}).get("distance_norm_0to1", np.nan))
        )
        metrics["harmony_chroma_dtw"].append(
            safe_float(h.get("chroma_similarity", {}).get("chroma_dtw_cosine", np.nan))
        )
        metrics["harmony_chroma_mean"].append(
            safe_float(h.get("chroma_similarity", {}).get("mean_chroma_cosine", np.nan))
        )
        metrics["harmony_best_shift_chroma_dtw"].append(
            safe_float(h.get("chroma_similarity", {}).get("best_shift_chroma_dtw_cosine", np.nan))
        )
        metrics["harmony_best_shift_chroma_mean"].append(
            safe_float(h.get("chroma_similarity", {}).get("best_shift_mean_chroma_cosine", np.nan))
        )
        metrics["harmony_chord_root"].append(
            safe_float(h.get("chord_similarity", {}).get("mir_root", np.nan))
        )
        metrics["harmony_chord_majmin"].append(
            safe_float(h.get("chord_similarity", {}).get("mir_majmin", np.nan))
        )

        # ---------------- rhythm ----------------
        metrics["rhythm_delta_bpm"].append(
            safe_float(rm.get("delta_bpm", np.nan))
        )
        metrics["rhythm_delta_bpm_folded"].append(
            safe_float(rm.get("delta_bpm_folded", np.nan))
        )
        metrics["rhythm_beat_fmeasure"].append(
            safe_float(rm.get("beat_mir_eval", {}).get("F-measure", np.nan))
        )
        metrics["rhythm_info_gain"].append(
            safe_float(rm.get("beat_mir_eval", {}).get("Information gain", np.nan))
        )

        # ---------------- structure ----------------
        metrics["structure_boundary_f"].append(
            safe_float(s.get("boundary_f_after_shift", s.get("boundary_f", np.nan)))
        )
        metrics["structure_ari"].append(
            safe_float(s.get("ari", np.nan))
        )
        metrics["structure_score"].append(
            safe_float(s.get("structural_form_score", np.nan))
        )

        # ---------------- melody raw ----------------
        m_raw = m.get("raw", {})
        metrics["melody_raw_overall_acc"].append(
            safe_float(m_raw.get("mir_melody", {}).get("overall_accuracy", np.nan))
        )
        metrics["melody_raw_voicing_recall"].append(
            safe_float(m_raw.get("mir_melody", {}).get("voicing_recall", np.nan))
        )
        metrics["melody_raw_pitch_acc"].append(
            safe_float(m_raw.get("mir_melody", {}).get("raw_pitch_accuracy", np.nan))
        )
        metrics["melody_raw_chroma_acc"].append(
            safe_float(m_raw.get("mir_melody", {}).get("raw_chroma_accuracy", np.nan))
        )
        metrics["melody_raw_contour_dtw"].append(
            safe_float(m_raw.get("contour_dtw", {}).get("pitch_dtw_cosine", np.nan))
        )
        metrics["melody_raw_motif_jaccard"].append(
            safe_float(m_raw.get("motif_overlap_n3", {}).get("jaccard", np.nan))
        )
        metrics["melody_raw_motif_recall"].append(
            safe_float(m_raw.get("motif_overlap_n3", {}).get("recall", np.nan))
        )

        # ---------------- melody normalized ----------------
        m_norm = m.get("normalized", {})
        metrics["melody_norm_overall_acc"].append(
            safe_float(m_norm.get("mir_melody", {}).get("overall_accuracy", np.nan))
        )
        metrics["melody_norm_voicing_recall"].append(
            safe_float(m_norm.get("mir_melody", {}).get("voicing_recall", np.nan))
        )
        metrics["melody_norm_pitch_acc"].append(
            safe_float(m_norm.get("mir_melody", {}).get("raw_pitch_accuracy", np.nan))
        )
        metrics["melody_norm_chroma_acc"].append(
            safe_float(m_norm.get("mir_melody", {}).get("raw_chroma_accuracy", np.nan))
        )
        metrics["melody_norm_contour_dtw"].append(
            safe_float(m_norm.get("contour_dtw", {}).get("pitch_dtw_cosine", np.nan))
        )
        metrics["melody_norm_motif_jaccard"].append(
            safe_float(m_norm.get("motif_overlap_n3", {}).get("jaccard", np.nan))
        )
        metrics["melody_norm_motif_recall"].append(
            safe_float(m_norm.get("motif_overlap_n3", {}).get("recall", np.nan))
        )

    return metrics


metrics = extract_metrics(all_results)

# =========================================================
# 4) REPORT
# =========================================================

EXPECTED = {
    "harmony_key_distance": "high / near 1",
    "harmony_chroma_dtw": "drop",
    "harmony_chroma_mean": "drop",
    "harmony_best_shift_chroma_dtw": "stay higher",
    "harmony_best_shift_chroma_mean": "stay higher",
    "harmony_chord_root": "drop",
    "harmony_chord_majmin": "drop",

    "rhythm_delta_bpm": "near 0",
    "rhythm_delta_bpm_folded": "near 0",
    "rhythm_beat_fmeasure": "high",
    "rhythm_info_gain": "high",

    "structure_boundary_f": "high",
    "structure_ari": "high-ish",
    "structure_score": "high-ish",

    "melody_raw_overall_acc": "mixed",
    "melody_raw_voicing_recall": "high",
    "melody_raw_pitch_acc": "drop",
    "melody_raw_chroma_acc": "drop for tritone",
    "melody_raw_contour_dtw": "high",
    "melody_raw_motif_jaccard": "low-mid",
    "melody_raw_motif_recall": "low-mid",

    "melody_norm_overall_acc": "recover",
    "melody_norm_voicing_recall": "high",
    "melody_norm_pitch_acc": "recover",
    "melody_norm_chroma_acc": "recover",
    "melody_norm_contour_dtw": "high",
    "melody_norm_motif_jaccard": "recover some",
    "melody_norm_motif_recall": "recover some",
}

def print_metric_block(title, keys, metrics_dict):
    print(f"\n{title}")
    print("-" * 112)
    print(f"{'Metric':<40} {'Median':>10} {'Std':>10} {'N':>6} {'Expected':>20}")
    print("-" * 112)

    for name in keys:
        med, std, n = summarize(metrics_dict[name])
        exp = EXPECTED.get(name, "")
        if n > 0:
            print(f"{name:<40} {med:>10.4f} {std:>10.4f} {n:>6} {exp:>20}")
        else:
            print(f"{name:<40} {'N/A':>10} {'N/A':>10} {0:>6} {exp:>20}")

print("\n" + "=" * 112)
print(f"PITCH SHIFT (+{PITCH_SHIFT_STEPS} semitones) — {len(all_results)} attempted pairs, {len(metrics['harmony_key_distance'])} valid")
print("=" * 112)

print_metric_block(
    "HARMONY (should change)",
    [
        "harmony_key_distance",
        "harmony_chroma_dtw",
        "harmony_chroma_mean",
        "harmony_best_shift_chroma_dtw",
        "harmony_best_shift_chroma_mean",
        "harmony_chord_root",
        "harmony_chord_majmin",
    ],
    metrics,
)

print_metric_block(
    "RHYTHM (should remain largely stable)",
    [
        "rhythm_delta_bpm",
        "rhythm_delta_bpm_folded",
        "rhythm_beat_fmeasure",
        "rhythm_info_gain",
    ],
    metrics,
)

print_metric_block(
    "STRUCTURE (should remain largely stable)",
    [
        "structure_boundary_f",
        "structure_ari",
        "structure_score",
    ],
    metrics,
)

print_metric_block(
    "MELODY RAW (absolute-pitch-sensitive)",
    [
        "melody_raw_overall_acc",
        "melody_raw_voicing_recall",
        "melody_raw_pitch_acc",
        "melody_raw_chroma_acc",
        "melody_raw_contour_dtw",
        "melody_raw_motif_jaccard",
        "melody_raw_motif_recall",
    ],
    metrics,
)

print_metric_block(
    "MELODY NORMALIZED (transposition-corrected)",
    [
        "melody_norm_overall_acc",
        "melody_norm_voicing_recall",
        "melody_norm_pitch_acc",
        "melody_norm_chroma_acc",
        "melody_norm_contour_dtw",
        "melody_norm_motif_jaccard",
        "melody_norm_motif_recall",
    ],
    metrics,
)

print("\n" + "=" * 112)

valid_examples = [r for r in all_results if "error" not in r]
if valid_examples:
    print(json.dumps(valid_examples[0], indent=2))