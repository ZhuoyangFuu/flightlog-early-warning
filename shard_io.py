#!/usr/bin/env python
"""Single source of truth for shard encoding/decoding.

The storage dtype was previously hardcoded in three places (tensorize,
validate_tensors, ews_baseline). That drift is exactly how the float16
overflow bug survived: the writer and the checker agreed with each other and
were both wrong. Everything now imports from here.

Storage format, per log row:
    x     : float32, shape (n_bins, n_ch), C-order, zero where invalid
    mask  : np.packbits of the (n_bins, n_ch) bool validity array
    t_e_bin : int, -1 when the log has no primary trigger
"""
import numpy as np

# float16 is NOT safe for this corpus: vehicle_air_data.baro_pressure_pa is
# ~101325 Pa against an fp16 ceiling of 65504 (98% of the channel became +inf),
# and vehicle_local_position.x/y saturate at exactly 65504.
STORE_DTYPE = np.float32


def encode(x, valid):
    return {"x": x.astype(STORE_DTYPE).tobytes(),
            "mask": np.packbits(valid, axis=None).tobytes()}


def decode(row):
    """Return (x, valid) as (n_bins, n_ch) float32 and bool arrays."""
    n, c = row["n_bins"], row["n_ch"]
    x = np.frombuffer(row["x"], dtype=STORE_DTYPE).reshape(n, c)
    m = np.unpackbits(np.frombuffer(row["mask"], dtype=np.uint8))[: n * c]
    return x, m.reshape(n, c).astype(bool)
