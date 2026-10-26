"""Every number the pipeline shares, in one place.

Other modules import these constants instead of hard-coding values, so a
change here (say, a different window length) changes the whole pipeline
consistently. If the constants disagreed between modules, windows would be
built at one length and models would expect another — silent wrong answers.

Units and conventions:
- Sampling rates are in Hz. WIN/STEP/TRIAL_DROP are in *output* samples
  (after decimation to FS_OUT), except TRIAL_DROP_S which is seconds.
- Channel indices are 0-based Python indices into the 32 raw columns of a
  GrabMyo record. The PhysioNet description numbers them 1–32; we subtract
  one. FOREARM+WRIST+UNUSED partition the 32 columns exactly.
- Gesture labels stay 1-based as in the filenames (gesture1 … gesture17);
  17 is the rest class — see DECISIONS.md.
"""

FS_IN = 2048          # GrabMyo native sampling rate, Hz
FS_OUT = 1024         # after ×2 decimation, Hz
BAND = (20.0, 450.0)  # Butterworth band-pass edges, Hz
WIN = 256             # window length in FS_OUT samples (250 ms)
STEP = 128            # window step in FS_OUT samples (50% overlap)
TRIAL_DROP_S = 0.5    # seconds dropped at the start of each trial
N_GESTURES = 17       # 16 gestures + rest (rest = 17)

# Channel maps (0-based). 1-based in the dataset description:
#   forearm = 1..16, wrist = 18..23 and 26..31, unused = 17, 24, 25, 32.
FOREARM = list(range(0, 16))                      # desc cols 1–16
WRIST = list(range(17, 23)) + list(range(25, 31))  # desc cols 18–23, 26–31
UNUSED = [16, 23, 24, 31]                          # desc cols 17, 24, 25, 32
N_CHANNELS_RAW = 32
# Order stored in the memmap: forearm 16 first, then wrist 12.
KEEP = FOREARM + WRIST
N_CHANNELS = len(KEEP)  # 28

GESTURE_NAMES = [
    "Lateral Prehension", "Thumb Adduction",
    "Thumb and Little Finger Opposition", "Thumb and Index Finger Opposition",
    "Thumb and Index Finger Extension", "Thumb and Little Finger Extension",
    "Index and Middle Finger Extension", "Little Finger Extension",
    "Index Finger Extension", "Thumb Finger Extension",
    "Wrist Extension", "Wrist Flexion",
    "Forearm Supination", "Forearm Pronation",
    "Hand Open", "Hand Close", "Rest",
]  # index 0 = gesture 1 … index 16 = gesture 17 (Rest), per MotionSequence.txt

# Deterministic seed shared by anything stochastic.
SEED = 20261026
