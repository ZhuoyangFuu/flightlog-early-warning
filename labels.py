#!/usr/bin/env python
"""Weak-label extraction for UAV loss-of-control, revised against measured data.

WHY THIS DIFFERS FROM PROJECT_PLAN.md SECTION 5
-----------------------------------------------
Trigger 3 as originally written ("FailsafeFlags asserting a critical condition
while armed") is not implementable as stated. Measured on 34 sample logs:

    mode_req_*                 fired in 100% of logs
    auto_mission_missing       fired in  94%
    offboard_control_signal_lost fired in 91%

`mode_req_*` are static per-flight-mode CAPABILITY REQUIREMENTS, not failures.
Using failsafe_flags wholesale would label ~71% of all flights positive and
destroy the benchmark. Fields are therefore partitioned explicitly below.

A second measured hazard: failsafe_flags SCHEMA VARIES BY FIRMWARE. Of 54
observed fields, only ~30 appear in all logs; `fd_alt_loss`, `motor_damage`,
`slam_failed`, `collision_and_tilting` appeared in 1 of 34. A flag that "did
not fire" may simply be ABSENT. We record a per-log schema-presence mask so
that trigger rate is never confounded with firmware version -- otherwise
Split C (cross-firmware) is silently contaminated.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Trigger partition. Every failsafe_flags field observed is classified; unknown
# fields default to IGNORED and are reported, never silently treated as failures.
# ---------------------------------------------------------------------------

# Tier A: direct evidence of loss of control / airframe failure.
FF_PRIMARY = {
    "fd_critical_failure", "fd_motor_failure", "fd_esc_arming_failure",
    "fd_imbalanced_prop", "fd_alt_loss", "motor_damage",
    "collision_and_tilting", "attitude_invalid", "angular_velocity_invalid",
    "vtol_fixed_wing_system_failure",
}

# Tier B: degraded state. Often precedes LOC but is frequently benign on its
# own (a GPS dropout in position mode triggers a controlled RTL). Reported as
# a separate label stream; never merged into the primary positive set.
FF_SECONDARY = {
    "navigator_failure", "mission_failure", "geofence_breached",
    "primary_geofence_breached", "wind_limit_exceeded",
    "flight_time_limit_exceeded", "battery_unhealthy", "battery_too_hot",
    "battery_low_remaining_time", "parachute_unhealthy", "slam_failed",
    "innovation_hor_invalid", "odometry_position_invalid",
}

# Explicitly benign / configuration state. NEVER a positive.
FF_IGNORED = {
    "auto_mission_missing", "offboard_control_signal_lost", "battery_warning",
    "gcs_connection_lost", "manual_control_signal_lost", "gnss_lost",
    "global_position_invalid", "global_position_invalid_relaxed",
    "local_position_invalid", "local_position_invalid_relaxed",
    "local_velocity_invalid", "local_altitude_invalid", "home_position_invalid",
    "local_position_accuracy_low", "position_accuracy_low",
    "innovation_hor_not_good", "remote_id_unhealthy",
}

# failure_detector_status boolean fields (trigger 1). These are the cleanest
# signal in the corpus: PX4's own failure detector latching a fault.
FDS_PRIMARY = [
    "fd_roll", "fd_pitch", "fd_alt", "fd_ext", "fd_arm_escs",
    "fd_imbalanced_prop", "fd_battery",
]
FDS_IGNORED = ["fd_high_wind"]  # environmental, not a control failure

# PX4 nav_state values indicating an autopilot-commanded failsafe response.
NAV_FAILSAFE = {5: "AUTO_RTL", 18: "AUTO_LAND", 19: "AUTO_LANDGPSFAIL",
                20: "AUTO_DESCEND", 21: "TERMINATION"}

ARMING_STATE_ARMED = 2

# Channels that must never reach the model (Section 5 leakage control).
# Enforced in code, not by convention.
LEAKAGE_EXCLUDE = {
    "failure_detector_status", "failsafe_flags", "vehicle_status",
    "vehicle_status_flags", "estimator_status_flags", "vehicle_command",
    "vehicle_command_ack", "log_message", "mavlink_log",
    "position_controller_status", "navigator_mission_item",
}


def _topic(ulog, name):
    for d in ulog.data_list:
        if d.name == name:
            return d
    return None


def armed_intervals(ulog):
    """Return (t_start, t_end) seconds for each armed interval, from vehicle_status."""
    d = _topic(ulog, "vehicle_status")
    if d is None or "arming_state" not in d.data:
        return []
    ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
    st = np.asarray(d.data["arming_state"])
    order = np.argsort(ts, kind="stable")
    ts, st = ts[order], st[order]
    armed = st == ARMING_STATE_ARMED
    out, i = [], 0
    while i < len(armed):
        if armed[i]:
            j = i
            while j + 1 < len(armed) and armed[j + 1]:
                j += 1
            out.append((float(ts[i]), float(ts[j])))
            i = j + 1
        else:
            i += 1
    return out


def airborne_mask(ulog, t):
    """Boolean per-sample: vehicle is airborne (armed and not landed).

    Uses vehicle_land_detected when available, else falls back to armed-only.
    Gating on airborne matters: many flags assert on the ground pre-arm.
    """
    d = _topic(ulog, "vehicle_land_detected")
    if d is None or "landed" not in d.data:
        return None
    ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
    landed = np.asarray(d.data["landed"]).astype(bool)
    order = np.argsort(ts, kind="stable")
    ts, landed = ts[order], landed[order]
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    return ~landed[idx]


def extract_events(ulog):
    """Extract candidate loss-of-control events with provenance.

    Returns a dict with:
      primary   : list of {t, source, field} -- Tier A evidence
      secondary : list of {t, source, field} -- Tier B evidence
      t_e       : earliest primary event time while airborne, or None
      schema    : which trigger fields were PRESENT in this log's format
      armed     : armed intervals
    """
    res = {"primary": [], "secondary": [], "t_e": None,
           "schema": {"ff_present": [], "fds_present": [], "unknown_ff": []},
           "armed": armed_intervals(ulog), "notes": []}
    arm = res["armed"]
    if not arm:
        res["notes"].append("no_armed_interval")

    def in_armed(tv):
        return any(a <= tv <= b for a, b in arm)

    # ---- trigger 1: failure_detector_status --------------------------------
    d = _topic(ulog, "failure_detector_status")
    if d is not None:
        ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        air = airborne_mask(ulog, ts)
        present = {f.field_name for f in d.field_data}
        for fn in FDS_PRIMARY:
            if fn not in present:
                continue
            res["schema"]["fds_present"].append(fn)
            a = np.asarray(d.data[fn])[order]
            if a.dtype.kind not in "biu":
                continue
            fire = a > 0
            if air is not None:
                fire &= air
            idx = np.flatnonzero(fire)
            if len(idx):
                t0 = float(ts[idx[0]])
                if in_armed(t0):
                    res["primary"].append({"t": t0, "source": "failure_detector_status",
                                           "field": fn, "n": int(fire.sum())})

    # ---- trigger 3 (repaired): failsafe_flags ------------------------------
    d = _topic(ulog, "failsafe_flags")
    if d is not None:
        ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        air = airborne_mask(ulog, ts)
        for f in d.field_data:
            fn = f.field_name
            if fn == "timestamp":
                continue
            if fn in FF_PRIMARY:
                bucket = "primary"
            elif fn in FF_SECONDARY:
                bucket = "secondary"
            elif fn in FF_IGNORED or fn.startswith("mode_req_"):
                res["schema"]["ff_present"].append(fn)
                continue
            else:
                # unknown field: report, never treat as a failure
                res["schema"]["unknown_ff"].append(fn)
                continue
            res["schema"]["ff_present"].append(fn)
            a = np.asarray(d.data[fn])[order]
            if a.dtype.kind not in "biu":
                continue
            fire = a > 0
            if air is not None:
                fire &= air
            idx = np.flatnonzero(fire)
            if len(idx):
                t0 = float(ts[idx[0]])
                if in_armed(t0):
                    res[bucket].append({"t": t0, "source": "failsafe_flags",
                                        "field": fn, "n": int(fire.sum())})

    # ---- trigger 2: nav_state entering a failsafe mode while airborne ------
    d = _topic(ulog, "vehicle_status")
    if d is not None and "nav_state" in d.data:
        ts = np.asarray(d.data["timestamp"], dtype=np.float64) / 1e6
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        ns = np.asarray(d.data["nav_state"])[order]
        air = airborne_mask(ulog, ts)
        trans = np.flatnonzero(np.diff(ns.astype(np.int64)) != 0) + 1
        for i in trans:
            if int(ns[i]) in NAV_FAILSAFE and (air is None or air[i]):
                t0 = float(ts[i])
                if in_armed(t0):
                    res["secondary"].append({
                        "t": t0, "source": "nav_state",
                        "field": NAV_FAILSAFE[int(ns[i])], "n": 1})
                    break

    if res["primary"]:
        res["t_e"] = min(e["t"] for e in res["primary"])
    return res


def summarize(res):
    return {
        "t_e": res["t_e"],
        "n_primary": len(res["primary"]),
        "n_secondary": len(res["secondary"]),
        "primary_fields": sorted({e["field"] for e in res["primary"]}),
        # Per-event times: lets t_e be recomputed under a revised trigger set
        # without re-parsing raw (which no longer exists for bulk logs).
        "primary_events": [{"field": e["field"], "t": round(e["t"], 3)}
                           for e in sorted(res["primary"], key=lambda e: e["t"])],
        "secondary_fields": sorted({e["field"] for e in res["secondary"]}),
        "armed_s": round(sum(b - a for a, b in res["armed"]), 1) if res["armed"] else 0.0,
        "n_ff_fields": len(res["schema"]["ff_present"]),
        "n_fds_fields": len(res["schema"]["fds_present"]),
        "unknown_ff": sorted(set(res["schema"]["unknown_ff"])),
        "notes": res["notes"],
    }
