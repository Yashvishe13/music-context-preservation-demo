# utils.py

import numpy as np

# =========================================================
# HYPERPARAMETERS
# =========================================================

EPSILON = 1e-8
MAX_CIRCLE_STEPS = 6.0
MINOR_TO_MAJOR_OFFSET = 3
DTW_FALLBACK_DIAGONAL = True
COSINE_TO_SIMILARITY = True


# =========================================================
# CONSTANTS
# =========================================================

PITCH_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0
}

PC_TO_NAME = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

CIRCLE_OF_FIFTHS_INDEX = {
    0: 0,   # C
    7: 1,   # G
    2: 2,   # D
    9: 3,   # A
    4: 4,   # E
    11: 5,  # B
    6: 6,   # F#
    1: 7,   # C#
    8: 8,   # Ab
    3: 9,   # Eb
    10: 10, # Bb
    5: 11,  # F
}


# =========================================================
# FUNCTIONS
# =========================================================

def _relative_major_pc(pc: int, scale: str) -> int:
    if scale.lower().startswith("min"):
        return (pc + MINOR_TO_MAJOR_OFFSET) % 12
    return pc % 12


def circle_of_fifths_distance(key1: str, scale1: str, key2: str, scale2: str):
    pc1 = PITCH_TO_PC.get(key1)
    pc2 = PITCH_TO_PC.get(key2)

    if pc1 is None or pc2 is None:
        return None, None

    r1 = _relative_major_pc(pc1, scale1)
    r2 = _relative_major_pc(pc2, scale2)

    i1 = CIRCLE_OF_FIFTHS_INDEX[r1]
    i2 = CIRCLE_OF_FIFTHS_INDEX[r2]

    steps = abs(i1 - i2)
    steps = min(steps, 12 - steps)

    dist_norm = steps / MAX_CIRCLE_STEPS

    return steps, dist_norm


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)

    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0

    na = np.linalg.norm(a) + EPSILON
    nb = np.linalg.norm(b) + EPSILON

    return float(np.dot(a, b) / (na * nb))


def dtw_cosine(X: np.ndarray, Y: np.ndarray) -> float:
    from scipy.spatial.distance import cdist

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    if len(X) == 0 or len(Y) == 0:
        return 0.0

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + EPSILON)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + EPSILON)

    C = cdist(Xn, Yn, metric="cosine")
    C = np.nan_to_num(C, nan=1.0, posinf=1.0, neginf=1.0)

    try:
        import librosa
        _, wp = librosa.sequence.dtw(C=C)
        path = wp[::-1]
    except Exception:
        if DTW_FALLBACK_DIAGONAL:
            t = min(len(Xn), len(Yn))
            path = [(i, i) for i in range(t)]
        else:
            return 0.0

    if COSINE_TO_SIMILARITY:
        sims = [1.0 - C[i, j] for (i, j) in path]
    else:
        sims = [C[i, j] for (i, j) in path]

    return float(np.mean(sims)) if sims else 0.0


def _safe_float(x, default=np.nan):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default