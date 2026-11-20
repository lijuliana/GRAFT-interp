"""Fetch the small released files into memory: PPS vector (.pt, parsed with a
restricted unpickler, no torch), natural layer-20 activations (.npz)."""
import io, pickle, zipfile
import numpy as np
import st_remote as st

REV = "main"
HF = "https://huggingface.co/samuel500/graft-loreft-adapters/resolve/" + REV + "/"

class _Storage:
    def __init__(self, arr): self.arr = arr

def _rebuild_tensor_v2(storage, offset, size, stride, *rest):
    n = int(np.prod(size)) if len(size) else 1
    return storage.arr[offset: offset + n].reshape(size).copy()

class _DT:
    def __init__(self, dt): self.dt = dt

_ALLOWED = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch", "FloatStorage"): _DT("<f4"),
    ("torch", "HalfStorage"): _DT("<f2"),
    ("torch", "DoubleStorage"): _DT("<f8"),
    ("torch", "LongStorage"): _DT("<i8"),
    ("collections", "OrderedDict"): dict,
}

def load_pt(raw):
    zf = zipfile.ZipFile(io.BytesIO(raw))
    names = zf.namelist()
    pkl = [n for n in names if n.endswith("data.pkl")][0]
    root = pkl[: -len("data.pkl")]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if (mod, name) in _ALLOWED:
                return _ALLOWED[(mod, name)]
            raise pickle.UnpicklingError(f"blocked global {mod}.{name}")
        def persistent_load(self, pid):
            _, dt, key, _loc, _numel = pid
            return _Storage(np.frombuffer(zf.read(f"{root}data/{key}"), dtype=dt.dt))
    return U(io.BytesIO(zf.read(pkl))).load()

def pps():
    return load_pt(st._get(HF + "steering_vector/pps_vector_hackonly.pt"))

def natural():
    z = np.load(io.BytesIO(st._get(HF + "steering_vector/natural_h20.npz")), allow_pickle=False)
    return {k: z[k] for k in z.files}

if __name__ == "__main__":
    p = pps()
    print("pps keys", list(p.keys()), "layers", sorted(p["raw"].keys())[:3], "...", len(p["raw"]))
    print("meta", {k: (v if not hasattr(v, "shape") else v.shape) for k, v in p.get("meta", {}).items()})
    print("norm raw[20]", float(np.linalg.norm(p["raw"][20])))
    nat = natural()
    for k, v in nat.items(): print(k, v.shape, v.dtype)
    d = nat["dir_dn"]; u = p["raw"][20]
    print("cos(dir_dn, pps20)", float(d @ u / np.linalg.norm(d) / np.linalg.norm(u)))
    n, h = st.header(HF + "mlp-loreft/cdhf100_gr_m64_lf1_L20_k8_s9/model.safetensors")
    for k, v in h.items():
        if k != "__metadata__" and "retain" not in k: print(k, v["dtype"], v["shape"])
