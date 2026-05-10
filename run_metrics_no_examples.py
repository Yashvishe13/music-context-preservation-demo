import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
MCP_METRICS = os.path.join(ROOT, "mcp_metrics")
if MCP_METRICS not in sys.path:
    sys.path.insert(0, MCP_METRICS)

import numpy as np
np.int = int
np.float = float
np.complex = complex
np.bool = bool
np.object = object
np.str = str

from harmony_tonality import harmony_score
from rhythm_meter import rhythm_score
from structural_form import structural_score
from melody_motifs import melody_score_pitch_shift_aware

MUSIC = os.path.join(ROOT, "music")
ORIG_DIR = os.path.join(MUSIC, "original_audio")
EDITS_AUDIO_DIR = os.path.join(MUSIC, "edits_audio")

SEMI_RE = re.compile(r"([+-]?\d+)\s*semitone")


def slug_to_n_steps(slug):
    m = SEMI_RE.search(slug)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def evaluate_pair(ref, est, n_steps=0):
    return {
        "harmony_tonality": harmony_score(ref, est),
        "rhythm_meter": rhythm_score(ref, est),
        "structural_form": structural_score(ref, est),
        "melodic_content": melody_score_pitch_shift_aware(ref, est, n_steps=n_steps),
    }


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_existing(out_json):
    if not os.path.exists(out_json):
        return [], set()
    with open(out_json) as f:
        results = json.load(f)
    done = {(r["section"], r["slug"], r["song_id"]) for r in results if "section" in r}
    return results, done


def run_edits(out_json):
    if not os.path.isdir(EDITS_AUDIO_DIR):
        print(f"missing dir: {EDITS_AUDIO_DIR}")
        return []

    results, done = load_existing(out_json)
    if done:
        print(f"resuming — {len(done)} entries already done")

    # Walk edits_audio/{section}/{slug}/{source_id}.wav
    for section in sorted(os.listdir(EDITS_AUDIO_DIR)):
        section_dir = os.path.join(EDITS_AUDIO_DIR, section)
        if not os.path.isdir(section_dir):
            continue

        for slug in sorted(os.listdir(section_dir)):
            slug_dir = os.path.join(section_dir, slug)
            if not os.path.isdir(slug_dir):
                continue

            n_steps = slug_to_n_steps(slug)
            files = sorted(f for f in os.listdir(slug_dir) if f.endswith(".wav"))

            for fn in files:
                song_id = os.path.splitext(fn)[0]
                ref = os.path.join(ORIG_DIR, fn)
                est = os.path.join(slug_dir, fn)

                if not os.path.exists(ref):
                    print(f"missing ref for {section}/{slug}/{fn}: {ref}")
                    continue

                if (section, slug, song_id) in done:
                    print(f"skip {section}/{slug}/{fn}")
                    continue

                print(f"{section}/{slug}/{fn} (ref={fn}, n_steps={n_steps})")

                entry = {
                    "ref": ref,
                    "est": est,
                    "section": section,
                    "slug": slug,
                    "song_id": song_id,
                    "n_steps": n_steps,
                }

                try:
                    entry.update(evaluate_pair(ref, est, n_steps=n_steps))
                except Exception as exc:
                    entry["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  ! error: {entry['error']}")

                results.append(entry)
                save_json(out_json, results)

    print(f"done — {len(results)} pairs → {out_json}")
    return results


if __name__ == "__main__":
    run_edits(os.path.join(ROOT, "edits_results.json"))
