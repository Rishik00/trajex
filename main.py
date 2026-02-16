"""
run_experiments.py
──────────────────
Runs encoding + generation analysis for every enabled model × prompt pair
and saves metrics to results/{model_slug}__{pair_name}.npz.

Usage:
    python run_experiments.py            # skip existing results
    python run_experiments.py --rerun    # overwrite everything
"""

import os, gc, argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as cfg

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


# ── Naming ────────────────────────────────────────────────────────────────────
def model_slug(name: str) -> str:
    return name.replace("/", "_")

def result_path(mname: str, pair_name: str) -> str:
    return os.path.join(cfg.RESULTS_DIR, f"{model_slug(mname)}__{pair_name}.npz")


# ── Model loading / unloading ─────────────────────────────────────────────────
def load_model(name: str):
    print(f"  Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        output_hidden_states=True,
        output_attentions=True,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(DEVICE)
    mdl.eval()
    return mdl, tok

def unload_model(mdl):
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Forward pass ──────────────────────────────────────────────────────────────
def get_hidden_states(mdl, tok, prompt):
    """
    Returns:
        hs   : (L+1, S, H)  — embedding layer + L transformer layers
        attn : (L, S, S)    — mean-head attention per transformer layer
        strs : list[str]
        ids  : list[int]

    Note: hs has L+1 entries (includes embedding), attn has L entries.
    hs[1:] and attn[:] are aligned — both index transformer layer 0..L-1.
    """
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    ids    = inputs["input_ids"][0].tolist()
    strs   = [tok.decode([t]) for t in ids]

    with torch.inference_mode():
        out = mdl(**inputs)

    hs = torch.stack(out.hidden_states)[:, 0, :, :].cpu().float()  # (L+1, S, H)

    # out.attentions: tuple of L tensors, each (B, n_heads, S, S)
    # mean across heads → (L, S, S)
    attn = torch.stack([a[0].mean(dim=0) for a in out.attentions]).cpu().float()

    return hs, attn, strs, ids


# ── Generation ────────────────────────────────────────────────────────────────
def generate(mdl, tok, prompt):
    """
    Generate using model.generate with a hook to capture last-token
    hidden states at every layer for each generation step.

    Returns:
        step_reps : list[(L+1, H)]
        tok_strs  : list[str]
        text      : str
    """
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    inputs  = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    S0      = inputs["input_ids"].shape[1]   # prompt length

    # Storage filled by hooks
    _step_hs: list = []   # per step: list of (H,) tensors, one per layer
    _cur_step: list = [[]]  # mutable container so closure can append

    def make_hook(layer_idx):
        def hook(module, inp, out):
            # out is typically a tuple; first element is the hidden state (B, S, H)
            h = out[0][0, -1, :].detach().cpu().float()  # (H,)
            _cur_step[0].append(h)
        return hook

    # Register hooks on every transformer layer
    layers = list(mdl.model.layers)  # works for Qwen2, Llama-style models
    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]

    # Also capture embedding output (layer 0 of hidden_states)
    def embed_hook(module, inp, out):
        h = out[0, -1, :].detach().cpu().float() if out.dim() == 3 else out[-1].detach().cpu().float()
        _cur_step[0].insert(0, h)

    # model.model.embed_tokens output — capture via the full model forward hook
    # Simpler: just use output_hidden_states in generate and parse after
    for h in handles:
        h.remove()

    # ── cleaner approach: output_hidden_states=True in generate ──────────────
    do_sample = cfg.TEMPERATURE > 0
    gen_out = mdl.generate(
        **inputs,
        max_new_tokens=cfg.MAX_TOKENS,
        do_sample=do_sample,
        temperature=cfg.TEMPERATURE if do_sample else None,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    # gen_out.hidden_states: tuple of steps, each is tuple of L+1 tensors (B, S, H)
    # S grows each step; we only want the last position (newly generated token)
    step_reps = []
    for step_hs in gen_out.hidden_states:
        # step_hs: tuple of (L+1) tensors, each (B, S_current, H)
        layer_vecs = torch.stack(
            [step_hs[l][0, -1, :].cpu().float() for l in range(len(step_hs))]
        )  # (L+1, H)
        step_reps.append(layer_vecs)

    gen_ids  = gen_out.sequences[0, S0:].tolist()
    tok_strs = [tok.decode([t]) for t in gen_ids]
    text     = tok.decode(gen_ids, skip_special_tokens=True)

    return step_reps, tok_strs, text


# ── Metrics ───────────────────────────────────────────────────────────────────
def l2(a, b):
    return torch.norm(a - b, p=2).item()

def cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()

def l2_per_layer(ra, rb, n):
    return np.array([l2(ra[i], rb[i]) for i in range(n)])

def cos_per_layer(ra, rb, n):
    return np.array([cos(ra[i], rb[i]) for i in range(n)])

def turbulence(reps, n):
    """Directional consistency: cos(δ_i, δ_{i+1}) across layers."""
    deltas = [reps[i+1] - reps[i] for i in range(n-1)]
    return np.array([
        cos(deltas[i], deltas[i+1]) for i in range(len(deltas)-1)
    ])

def sign_flip(reps, n):
    """Fraction of dims that flip delta sign across consecutive layers."""
    deltas = [(reps[i+1] - reps[i]).numpy() for i in range(n-1)]
    ratios = []
    for i in range(len(deltas)-1):
        sa, sb = np.sign(deltas[i]), np.sign(deltas[i+1])
        nz     = (sa != 0) & (sb != 0)
        ratios.append(0.5 if nz.sum() == 0
                      else (sa[nz] != sb[nz]).sum() / nz.sum())
    return np.array(ratios)

def attn_divergence(attn):
    """
    Attention divergence per token per layer.

    attn : (L, S, S) — mean-head attention, attn[l, i, j] = how much
                        token i attends to token j at layer l.

    div[l, i] = col_sum[i] - row_sum[i]
              = (how much others attend to i) - (how much i attends to others)

    Positive → token is a sink (information flows in)
    Negative → token is a source (information flows out)

    Returns: (L, S)
    """
    col_sum = attn.sum(dim=1)   # (L, S) — total attention received by each token
    row_sum = attn.sum(dim=2)   # (L, S) — total attention sent by each token
    return (col_sum - row_sum).numpy()  # (L, S)


# ── Encoding analysis ─────────────────────────────────────────────────────────
def run_encoding(mdl, tok, pair, n_layers):
    hs_b, attn_b, toks_b, ids_b = get_hidden_states(mdl, tok, pair["benign"])
    hs_h, attn_h, toks_h, ids_h = get_hidden_states(mdl, tok, pair["harmful"])

    lr = n_layers + 1 - cfg.EXCLUDE_LAST_N   # usable layer range (hidden states)
    al = n_layers - cfg.EXCLUDE_LAST_N        # usable layer range (attention, no embed)

    # resolve token positions
    if cfg.TOKEN_POSITIONS is not None:
        seq_len   = min(hs_b.shape[1], hs_h.shape[1])
        positions = [seq_len + p if p < 0 else p for p in cfg.TOKEN_POSITIONS]
    else:
        mn  = min(len(ids_b), len(ids_h))
        div = [i for i in range(mn) if ids_b[i] != ids_h[i]]
        positions = sorted(set(div + [mn - 1]))

    n_pos = len(positions)
    L     = lr

    out = {
        "enc_positions": np.array(positions),
        "enc_toks_b":    np.array(toks_b, dtype=object),
        "enc_toks_h":    np.array(toks_h, dtype=object),
        # hidden-state metrics: (n_pos, L)
        "enc_l2":        np.zeros((n_pos, L)),
        "enc_cos":       np.zeros((n_pos, L)),
        "enc_l2_vel":    np.zeros((n_pos, L)),
        "enc_turb_b":    np.zeros((n_pos, L-2)),
        "enc_turb_h":    np.zeros((n_pos, L-2)),
        "enc_flip_b":    np.zeros((n_pos, L-2)),
        "enc_flip_h":    np.zeros((n_pos, L-2)),
        # attention divergence: (al, S) — same S for both (shared prompt prefix)
        "enc_div_b":     np.zeros((al, min(hs_b.shape[1], hs_h.shape[1]))),
        "enc_div_h":     np.zeros((al, min(hs_b.shape[1], hs_h.shape[1]))),
        "enc_div_vel_b": np.zeros((al, min(hs_b.shape[1], hs_h.shape[1]))),
        "enc_div_vel_h": np.zeros((al, min(hs_b.shape[1], hs_h.shape[1]))),
    }

    # hidden-state metrics
    for i, pos in enumerate(positions):
        rb  = hs_b[:lr, pos, :]
        rh  = hs_h[:lr, pos, :]
        l2v = l2_per_layer(rb, rh, lr)
        out["enc_l2"][i]     = l2v
        out["enc_cos"][i]    = cos_per_layer(rb, rh, lr)
        out["enc_l2_vel"][i] = np.gradient(l2v)
        out["enc_turb_b"][i] = turbulence(rb, lr)
        out["enc_turb_h"][i] = turbulence(rh, lr)
        out["enc_flip_b"][i] = sign_flip(rb, lr)
        out["enc_flip_h"][i] = sign_flip(rh, lr)

    # attention divergence — computed over all tokens, not just sel positions
    S = min(hs_b.shape[1], hs_h.shape[1])
    div_b = attn_divergence(attn_b[:al, :S, :S])   # (al, S)
    div_h = attn_divergence(attn_h[:al, :S, :S])   # (al, S)
    out["enc_div_b"]     = div_b
    out["enc_div_h"]     = div_h
    # rate of change across layers for each token
    out["enc_div_vel_b"] = np.gradient(div_b, axis=0)  # (al, S)
    out["enc_div_vel_h"] = np.gradient(div_h, axis=0)  # (al, S)

    return out


# ── Generation analysis ───────────────────────────────────────────────────────
def run_generation(mdl, tok, pair, n_layers):
    al = n_layers // 2

    reps_b, tstrs_b, text_b = generate(mdl, tok, pair["benign"])
    reps_h, tstrs_h, text_h = generate(mdl, tok, pair["harmful"])

    T = min(len(reps_b), len(reps_h))

    l2v  = np.array([l2(reps_b[t][al],  reps_h[t][al])  for t in range(T)])
    cosv = np.array([cos(reps_b[t][al], reps_h[t][al])  for t in range(T)])

    dir_stab = np.array([
        cos(reps_b[t][al] - reps_h[t][al],
            reps_b[t+1][al] - reps_h[t+1][al])
        for t in range(T-1)
    ])

    dturb_b, dturb_h = [], []
    for t in range(1, T-1):
        db_prev = reps_b[t][al]   - reps_b[t-1][al]
        db_cur  = reps_b[t+1][al] - reps_b[t][al]
        dh_prev = reps_h[t][al]   - reps_h[t-1][al]
        dh_cur  = reps_h[t+1][al] - reps_h[t][al]
        dturb_b.append(cos(db_prev, db_cur))
        dturb_h.append(cos(dh_prev, dh_cur))

    return {
        "gen_l2":             l2v,
        "gen_cos":            cosv,
        "gen_dir_stab":       dir_stab,
        "gen_turb_b":         np.array(dturb_b),
        "gen_turb_h":         np.array(dturb_h),
        "gen_toks_b":         np.array(tstrs_b[:T], dtype=object),
        "gen_toks_h":         np.array(tstrs_h[:T], dtype=object),
        "gen_text_b":         np.array(text_b),
        "gen_text_h":         np.array(text_h),
        "gen_analysis_layer": np.array(al),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(rerun: bool):
    for mname, mcfg in cfg.MODELS.items():
        if not mcfg.get("enabled", True):
            print(f"Skipping {mname} (disabled)")
            continue

        mdl, tok = load_model(mname)
        n_layers = mdl.config.num_hidden_layers
        print(f"  Layers: {n_layers}  |  Device: {DEVICE}")

        for pair in cfg.PAIRS:
            path = result_path(mname, pair["name"])
            if os.path.exists(path) and not rerun:
                print(f"  [{pair['name']}] exists — skipping. Use --rerun to overwrite.")
                continue

            print(f"  Running pair: {pair['name']}")
            enc = run_encoding(mdl, tok, pair, n_layers)
            gen = run_generation(mdl, tok, pair, n_layers)

            meta = {
                "model":    np.array(mname),
                "pair":     np.array(pair["name"]),
                "benign":   np.array(pair["benign"]),
                "harmful":  np.array(pair["harmful"]),
                "n_layers": np.array(n_layers),
                "seed":     np.array(cfg.SEED),
            }

            np.savez(path, **meta, **enc, **gen)
            print(f"  Saved → {path}")

        unload_model(mdl)
        print()

    print("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="Overwrite existing results")
    args = parser.parse_args()
    main(args.rerun)