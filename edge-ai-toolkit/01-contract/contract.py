"""
 GENERATED FILE -- DO NOT EDIT
 Source: data_contract.py   Contract: DC-001 r1.0   Hash: d5a07fe0a05f

 Editing this by hand puts it out of sync with its
 counterpart in the other language, which is exactly the
 failure this file exists to prevent. Edit data_contract.py
 and regenerate ALL of them.
"""

CONTRACT_HASH = "d5a07fe0a05f"

# --- sampling ---
SAMPLE_RATE_HZ        = 50
WINDOW_MS             = 2000
WINDOW_SAMPLES        = 100
SAMPLE_INTERVAL_MS    = 20.0

# --- channels: ORDER IS LOAD-BEARING ---
CHANNELS              = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
N_CHANNELS            = 6

# --- feature layout ---
STATS                 = ['mean', 'std', 'min', 'max']
N_STATS               = 4
N_FEATURES            = 24
FEATURE_LAYOUT        = 'channel_major'
STD_DDOF              = 0

# --- units ---
ACCEL_TICKS_PER_G     = 4096.0
GYRO_TICKS_PER_DPS    = 16.384

# --- sentinel: a value the sensor cannot produce ---
SENTINEL_MISSING      = -32768.0

# --- golden vector tolerances ---
GOLDEN_REL_TOL        = 0.0001
GOLDEN_ABS_TOL        = 0.001

# --- NOT YET DECIDED. None on purpose; see data_contract.py ---
DETECTION_THRESHOLD   = None
ALERT_HOLD_S          = None


def feature_index(channel, stat):
    """
    Index of one (channel, stat) pair in the feature vector.

    Both languages compute this the same way from the same
    layout constant, so neither can hand-roll it differently.
    """
    ch = CHANNELS.index(channel) if isinstance(channel, str) else channel
    st = STATS.index(stat) if isinstance(stat, str) else stat
    if FEATURE_LAYOUT == "channel_major":
        return ch * N_STATS + st
    return st * N_CHANNELS + ch
