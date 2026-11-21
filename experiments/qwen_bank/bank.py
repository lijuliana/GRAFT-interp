"""Sample and grade Countdown-Code rollouts from a released GRAFT checkpoint.

Run from the rl/ directory of samuel500/GRAFT (branch loreft) with
GRAFT_COUNTDOWN_SFT_MODEL and RH_REPO_PATH set. The generation and grading
recipe follows that repo's experiments/steering_variance/steering_variance.py
(phase `rollouts`): eval-split questions, all hackable, neutral prompt,
temperature 1, no top-k, 1536 new tokens, graded with countdown_all_components.

    python bank.py --ckpt ~/adapters/graft/<run> --scales 1,0 --q_start 0 --n_q 64 --k 16 --out bank/x.json

--forget_ckpt takes the forget adapter from a different run (the transplant test).
"""
import argparse, json, os, sys, time

import torch
import transformers

sys.path.insert(0, os.getcwd())

# Load the base model in bfloat16 straight away: the default would need 33 GB of RAM.
_orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained


def _bf16_from_pretrained(*a, **k):
    k.setdefault("torch_dtype", torch.bfloat16)
    k.setdefault("low_cpu_mem_usage", True)
    return _orig_from_pretrained(*a, **k)


transformers.AutoModelForCausalLM.from_pretrained = _bf16_from_pretrained


def load_model(ckpt, forget_ckpt=None):
    """Checkpoint on the frozen SFT base. With forget_ckpt, the forget adapter's weights come from that run."""
    from eval_utils import load_gradient_routing_model
    base = os.environ["GRAFT_COUNTDOWN_SFT_MODEL"]
    model = load_gradient_routing_model(os.path.expanduser(ckpt), base_model=base)
    if forget_ckpt:
        from safetensors.torch import load_file
        sd = load_file(os.path.join(os.path.expanduser(forget_ckpt), "model.safetensors"))
        sub = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items() if "_forget" in k}
        own = dict(model.named_parameters())
        assert sub and all(k in own for k in sub), "forget keys do not match the model"
        missing, unexpected = model.load_state_dict(sub, strict=False)
        assert not unexpected
        print(f"transplanted {len(sub)} forget tensors from {forget_ckpt}")
    return model.to(torch.bfloat16).cuda().eval()


def strip(ids, stop):
    out = []
    for t in ids:
        if int(t) in stop:
            break
        out.append(int(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--forget_ckpt", default=None)
    ap.add_argument("--scales", default="1,1", help="retain,forget")
    ap.add_argument("--q_start", type=int, default=0)
    ap.add_argument("--n_q", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gen_batch", type=int, default=2, help="questions per generate call (x k sequences)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        print(a.out, "exists, skipping"); return

    from transformers import AutoTokenizer, GenerationConfig
    from envs.countdown_code import _EVAL_START, _rows
    from gradient_routing import set_scales
    from rewards import _CC_SLOTS, countdown_all_components

    base = os.environ["GRAFT_COUNTDOWN_SFT_MODEL"]
    rs, fs = (float(x) for x in a.scales.split(","))
    rows = _rows("eval", a.q_start + a.n_q, _EVAL_START, 1.0)[a.q_start:]
    assert len(rows) == a.n_q and all(r["hackable"] for r in rows)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    prompt_ids = [tok(tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False), add_special_tokens=False)["input_ids"] for r in rows]
    model = load_model(a.ckpt, a.forget_ckpt)
    set_scales(model, rs, fs)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    cfg = GenerationConfig(max_new_tokens=1536, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                           eos_token_id=eos, pad_token_id=tok.pad_token_id)
    stop = set(eos) | {tok.pad_token_id}
    torch.manual_seed(a.seed)
    t0 = time.time(); recs = []
    for i in range(0, len(rows), a.gen_batch):
        enc = tok.pad([{"input_ids": ids} for ids in prompt_ids[i:i + a.gen_batch]], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, generation_config=cfg, num_return_sequences=a.k)
        new = gen[:, enc["input_ids"].shape[1]:].tolist()
        for j in range(enc["input_ids"].shape[0]):
            r = rows[i + j]
            for si, ids in enumerate(new[j * a.k:(j + 1) * a.k]):
                fin = "stop" if any(t in stop for t in ids) else "length"
                ids = strip(ids, stop)
                recs.append({"q_idx": a.q_start + i + j, "sample_idx": si, "row_id": r["id"], "nums": r["nums"],
                             "target": r["target"], "prompt": r["prompt"], "prompt_token_ids": prompt_ids[i + j],
                             "response_token_ids": ids, "finish": fin,
                             "completion": tok.decode(ids, skip_special_tokens=True)})
        print(f"[bank] {min(i + a.gen_batch, len(rows))}/{len(rows)} questions ({time.time() - t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache()
    comp = countdown_all_components([x["completion"] for x in recs], nums=[x["nums"] for x in recs],
                                    target=[x["target"] for x in recs], hackable=[True] * len(recs))
    for n, name in enumerate(_CC_SLOTS):
        for x, v in zip(recs, comp[n]):
            x[name] = float(v)
    n = len(recs)
    stats = {name: round(sum(x[name] for x in recs) / n, 4) for name in _CC_SLOTS}
    meta = {"ckpt": a.ckpt, "forget_ckpt": a.forget_ckpt, "scales": [rs, fs], "q_start": a.q_start, "n_q": a.n_q,
            "k": a.k, "seed": a.seed, "gen_seconds": round(time.time() - t0), "slots": list(_CC_SLOTS), "stats": stats}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"meta": meta, "rollouts": recs}, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    print(f"[bank] {n} rollouts in {meta['gen_seconds']}s: {json.dumps(stats)}")


if __name__ == "__main__":
    main()
