"""Read safetensors files over HTTP into memory (nothing written to disk)."""
import json, struct, urllib.request
import numpy as np

UA = {"User-Agent": "generic-client"}

def _get(url, start=None, end=None, tries=5):
    import time
    h = dict(UA)
    if start is not None:
        h["Range"] = f"bytes={start}-{end}"
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=90) as r:
                return r.read()
        except Exception as e:  # stalled or dropped connection: retry
            last = e; time.sleep(2 + 3 * i)
    raise last

def header(url):
    n = struct.unpack("<Q", _get(url, 0, 7))[0]
    return n, json.loads(_get(url, 8, 8 + n - 1))

def _decode(buf, dtype, shape):
    if dtype == "F32":
        a = np.frombuffer(buf, dtype="<f4")
    elif dtype == "F16":
        a = np.frombuffer(buf, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        a = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        raise ValueError(dtype)
    return a.reshape(shape)

def load_all(url):
    raw = _get(url)
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        out[k] = _decode(raw[base + s: base + e], v["dtype"], v["shape"])
    return out

def load_tensor(url, name, hdr=None, n=None):
    if hdr is None:
        n, hdr = header(url)
    v = hdr[name]
    s, e = v["data_offsets"]
    return _decode(_get(url, 8 + n + s, 8 + n + e - 1), v["dtype"], v["shape"])
