#!/usr/bin/env python
"""Canonical channel vocabulary for the FlightLog benchmark.

Every channel is a (topic, field) pair with a STABLE integer id. The id is what
the channel-identity embedding indexes, so it must never be reordered once
pretraining has begun -- append only.

Selection rules applied to the measured field survey:
  * keep physically meaningful continuous signals
  * drop device ids, serial numbers, calibration/reset counters, timestamps
  * drop bitfield flags (control_mode_flags, filter_fault_flags, health_flags,
    timeout_flags, solution_status_flags): these are autopilot fault state and
    sit too close to the label. Excluded as leakage, not for lack of signal.
  * drop fields present in <20% of logs (fake_*, terr_flow, jerk, zxy_*)
"""

# (topic, field) in stable order. DO NOT REORDER -- append only.
CHANNELS = [
    # --- attitude ---
    ("vehicle_attitude", "q[0]"), ("vehicle_attitude", "q[1]"),
    ("vehicle_attitude", "q[2]"), ("vehicle_attitude", "q[3]"),
    ("vehicle_attitude_setpoint", "q_d[0]"), ("vehicle_attitude_setpoint", "q_d[1]"),
    ("vehicle_attitude_setpoint", "q_d[2]"), ("vehicle_attitude_setpoint", "q_d[3]"),
    ("vehicle_attitude_setpoint", "thrust_body[2]"),
    # --- body rates ---
    ("vehicle_angular_velocity", "xyz[0]"), ("vehicle_angular_velocity", "xyz[1]"),
    ("vehicle_angular_velocity", "xyz[2]"),
    ("vehicle_rates_setpoint", "roll"), ("vehicle_rates_setpoint", "pitch"),
    ("vehicle_rates_setpoint", "yaw"),
    # --- local position / velocity ---
    ("vehicle_local_position", "x"), ("vehicle_local_position", "y"),
    ("vehicle_local_position", "z"), ("vehicle_local_position", "vx"),
    ("vehicle_local_position", "vy"), ("vehicle_local_position", "vz"),
    ("vehicle_local_position", "ax"), ("vehicle_local_position", "ay"),
    ("vehicle_local_position", "az"), ("vehicle_local_position", "heading"),
    ("vehicle_local_position", "dist_bottom"), ("vehicle_local_position", "eph"),
    ("vehicle_local_position", "epv"), ("vehicle_local_position", "evh"),
    ("vehicle_local_position", "evv"),
    ("vehicle_local_position_setpoint", "x"), ("vehicle_local_position_setpoint", "y"),
    ("vehicle_local_position_setpoint", "z"), ("vehicle_local_position_setpoint", "vx"),
    ("vehicle_local_position_setpoint", "vy"), ("vehicle_local_position_setpoint", "vz"),
    # --- raw IMU ---
    ("sensor_combined", "gyro_rad[0]"), ("sensor_combined", "gyro_rad[1]"),
    ("sensor_combined", "gyro_rad[2]"),
    ("sensor_combined", "accelerometer_m_s2[0]"),
    ("sensor_combined", "accelerometer_m_s2[1]"),
    ("sensor_combined", "accelerometer_m_s2[2]"),
    # --- actuators ---
    ("actuator_motors", "control[0]"), ("actuator_motors", "control[1]"),
    ("actuator_motors", "control[2]"), ("actuator_motors", "control[3]"),
    ("actuator_motors", "control[4]"), ("actuator_motors", "control[5]"),
    ("actuator_motors", "control[6]"), ("actuator_motors", "control[7]"),
    ("actuator_outputs", "output[0]"), ("actuator_outputs", "output[1]"),
    ("actuator_outputs", "output[2]"), ("actuator_outputs", "output[3]"),
    ("actuator_outputs", "output[4]"), ("actuator_outputs", "output[5]"),
    ("actuator_outputs", "output[6]"), ("actuator_outputs", "output[7]"),
    # --- estimator health (continuous ratios only, no flag bitfields) ---
    ("estimator_status", "vel_test_ratio"), ("estimator_status", "pos_test_ratio"),
    ("estimator_status", "hgt_test_ratio"), ("estimator_status", "tas_test_ratio"),
    ("estimator_status", "hagl_test_ratio"), ("estimator_status", "beta_test_ratio"),
    ("estimator_status", "mag_test_ratio"),
    ("estimator_status", "pos_horiz_accuracy"), ("estimator_status", "pos_vert_accuracy"),
    # --- estimator innovations ---
    ("estimator_innovations", "gps_hvel[0]"), ("estimator_innovations", "gps_hvel[1]"),
    ("estimator_innovations", "gps_vvel"),
    ("estimator_innovations", "gps_hpos[0]"), ("estimator_innovations", "gps_hpos[1]"),
    ("estimator_innovations", "gps_vpos"), ("estimator_innovations", "baro_vpos"),
    ("estimator_innovations", "heading"),
    ("estimator_innovations", "mag_field[0]"), ("estimator_innovations", "mag_field[1]"),
    ("estimator_innovations", "mag_field[2]"),
    # --- power / air / compute ---
    ("battery_status", "voltage_v"), ("battery_status", "current_a"),
    ("battery_status", "remaining"), ("battery_status", "temperature"),
    ("vehicle_air_data", "baro_alt_meter"), ("vehicle_air_data", "baro_pressure_pa"),
    ("vehicle_air_data", "rho"),
    ("cpuload", "load"), ("cpuload", "ram_usage"),
]

CHANNEL_ID = {c: i for i, c in enumerate(CHANNELS)}
N_CHANNELS = len(CHANNELS)
TOPICS = sorted({t for t, _ in CHANNELS})

# Fields per topic, for fast lookup during parsing.
TOPIC_FIELDS = {}
for _t, _f in CHANNELS:
    TOPIC_FIELDS.setdefault(_t, []).append(_f)

# ---------------------------------------------------------------------------
# Physical plausibility bounds, applied as |x| > bound -> MASK (not clip).
#
# Some source logs contain corrupt values: a max |accelerometer_m_s2[2]| of
# 4.2e32 was measured in the curated set, alongside pyulog "file corruption
# detected" warnings. These are rare (0.03% of bounded cells) but they would
# destroy any normalisation or gradient computed over them. We mask rather than
# clip, because a corrupt sample is missing data, not a large measurement.
#
# Bounds are deliberately generous -- several times any physically achievable
# value -- so that real extreme events (which are the whole point of this
# benchmark) are never masked. Keyed by (topic, field) prefix.
# ---------------------------------------------------------------------------
PLAUSIBLE_ABS = {
    "sensor_combined.accelerometer_m_s2": 500.0,    # ~50 g
    "sensor_combined.gyro_rad": 200.0,              # ~32 rev/s
    "vehicle_angular_velocity.xyz": 200.0,
    "vehicle_attitude.q": 1.05,                     # unit quaternion
    "vehicle_attitude_setpoint.q_d": 1.05,
    "vehicle_attitude_setpoint.thrust_body": 10.0,
    "vehicle_rates_setpoint": 200.0,
    "vehicle_local_position.x": 1.0e6,              # 1000 km from origin
    "vehicle_local_position.y": 1.0e6,
    "vehicle_local_position.z": 1.0e5,
    "vehicle_local_position.v": 500.0,              # vx, vy, vz
    "vehicle_local_position.a": 500.0,              # ax, ay, az
    "vehicle_local_position.heading": 10.0,         # radians
    "vehicle_local_position.dist_bottom": 1.0e4,
    "vehicle_local_position.e": 1.0e5,              # eph, epv, evh, evv
    "vehicle_local_position_setpoint": 1.0e6,
    "sensor_combined": 1.0e4,                       # catch-all for the topic
    "actuator_motors.control": 10.0,
    "actuator_outputs.output": 1.0e5,               # PWM or normalised
    "estimator_status": 1.0e5,
    "estimator_innovations": 1.0e5,
    "battery_status.voltage_v": 200.0,
    "battery_status.current_a": 2000.0,
    "battery_status.remaining": 2.0,
    "battery_status.temperature": 1000.0,
    "vehicle_air_data.baro_alt_meter": 1.0e5,
    "vehicle_air_data.baro_pressure_pa": 2.0e5,
    "vehicle_air_data.rho": 100.0,
    "cpuload.load": 2.0,
    "cpuload.ram_usage": 2.0,
}


def plausible_bound(topic, field):
    """Longest-prefix match for a channel's plausibility bound, or None."""
    key = f"{topic}.{field}"
    best, best_len = None, -1
    for k, v in PLAUSIBLE_ABS.items():
        if key.startswith(k) and len(k) > best_len:
            best, best_len = v, len(k)
    return best


# Per-channel bound array, indexed by channel id (inf where unbounded).
BOUNDS = [plausible_bound(t, f) for t, f in CHANNELS]


# ---------------------------------------------------------------------------
# CHANNEL GROUPS -- for the 2026-09-02 Tier-0 aggregation-width diagnostic.
#
# Why this exists. van Beers et al. (arXiv:2512.20868 / PNAS 2026) obtain a
# POSITIVE classical-EWS result for quadrotor loss of control using lag-1
# autocorrelation on ROTOR SPEED ALONE -- one signal, chosen by physics. Our
# Tier-0 aggregates a top-k z-score over all 87 channels and detects 18.6%.
# Ditlevsen & Ditlevsen (arXiv:2609.01164) show an early-warning signal that
# lives in a subspace can be destroyed by aggregating over the full state.
#
# So the open question a reviewer will ask is: is our negative a property of
# CLASSICAL EWS, or of OUR AGGREGATION? These groups let `ews_baseline.py`
# recompute Tier-0 over a physically-chosen subset and answer it.
#
# Definitions are frozen here BEFORE the diagnostic runs (this project's
# standing rule: declare the grid before reading any number from it).
#
# NOTE ON THE ROTOR ANALOGUE: PX4 public logs do not carry measured rotor speed
# for most vehicles (esc_status.esc_rpm is rarely present, which is why it is
# not in CHANNELS). `actuator_outputs.output[*]` is the COMMANDED rotor signal
# and is the closest available analogue -- van Beers' flyaway is described as
# "rapid and unstable oscillations in rotor speeds", which a closed loop drives
# through exactly this command. State the substitution whenever the result is
# quoted; it is a limitation, not an equivalence.
# ---------------------------------------------------------------------------
CHANNEL_GROUPS = {
    # the frozen Tier-0 channel set -- the control arm, must reproduce 0.186
    "all": None,
    # commanded rotor signal only: the direct van Beers analogue
    "rotor_out": ["actuator_outputs.output"],
    # both actuator representations (normalised control + PWM output)
    "rotor": ["actuator_motors.control", "actuator_outputs.output"],
    # the fast inner loop: body rates, measured and raw
    "rates": ["vehicle_angular_velocity.xyz", "sensor_combined.gyro_rad"],
    # rotor + rates: the signals a control-theoretic reading says carry the
    # instability, and nothing else
    "rotor_rates": ["actuator_motors.control", "actuator_outputs.output",
                    "vehicle_angular_velocity.xyz", "sensor_combined.gyro_rad"],
}


def channel_indices(spec):
    """Resolve a group name or comma-separated prefix list -> sorted channel ids.

    `spec` is either a key of CHANNEL_GROUPS or a comma-separated list of
    "topic.field" PREFIXES (e.g. "actuator_outputs.output,cpuload"). Returns
    None for the full vocabulary, so callers can skip slicing entirely.
    Raises ValueError if a spec matches nothing -- a silent empty channel set
    would produce an all-NaN score and look like a legitimate negative result.
    """
    if spec is None or spec == "all":
        return None
    prefixes = CHANNEL_GROUPS.get(spec)
    if prefixes is None:
        prefixes = [p.strip() for p in spec.split(",") if p.strip()]
    idx = [i for i, (t, f) in enumerate(CHANNELS)
           if any(f"{t}.{f}".startswith(p) for p in prefixes)]
    if not idx:
        raise ValueError(f"channel spec {spec!r} matched no channel")
    return idx


def channel_names(idx=None):
    """Canonical "topic.field" names, all of them or the given ids."""
    ids = range(N_CHANNELS) if idx is None else idx
    return [f"{CHANNELS[i][0]}.{CHANNELS[i][1]}" for i in ids]

TARGET_HZ = 20.0
BIN_S = 1.0 / TARGET_HZ

# Max staleness for forward-filling a slow channel before it is masked invalid.
# cpuload runs at 2 Hz, battery at 5 Hz, so 0.75 s covers legitimate cadence
# without inventing data across a real gap.
MAX_STALE_S = 0.75

if __name__ == "__main__":
    print(f"{N_CHANNELS} channels across {len(TOPICS)} topics")
    for t in TOPICS:
        print(f"  {t:34s} {len(TOPIC_FIELDS[t]):3d}")
