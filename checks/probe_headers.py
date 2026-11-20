"""Save the base model's safetensors header (tensor names and byte offsets)."""
import sys, json
import st_remote as st
A = "https://huggingface.co/samuel500/graft-repro-adapters/resolve/main/graft/cdhf100_gr_lccoh64_lr3_s9/model.safetensors"
n, h = st.header(A)
keys = [k for k in h if k != "__metadata__"]
print(len(keys), h.get("__metadata__"))
for k in keys[:8]: print(k, h[k]["dtype"], h[k]["shape"])
B = "https://huggingface.co/samuel500/graft-qwen3-8b-countdown-sft/resolve/a2d54fdf0aa9fa6ee8e487777ee7d63adaf8ec48/model.safetensors"
n2, h2 = st.header(B)
ks = [k for k in h2 if k != "__metadata__"]
print(len(ks))
for k in ks:
    if "embed" in k or "lm_head" in k or k.startswith("model.norm") or "layers.21." in k:
        print(k, h2[k]["dtype"], h2[k]["shape"])
json.dump({"n": n2, "hdr": h2}, open(sys.argv[1], "w"))
