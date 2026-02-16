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
from scipy.ndimage import uniform_filter1d

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
    """Returns (L+1, S, H), token strings, token ids."""
    inputs = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    ids    = inputs["input_ids"][0].tolist()
    strs   = [tok.decode([t]) for t in ids]
    with torch.inference_mode():
        out = mdl(**inputs)
        hs  = torch.stack(out.hidden_states)       # (L+1, B, S, H)
    return hs[:, 0, :, :].cpu().float(), strs, ids


# ── Generation ────────────────────────────────────────────────────────────────
def generate(mdl, tok, prompt):
    """
    Returns:
        step_reps : list[(L+1, H)]  — last-position hidden states per step
        tok_strs  : list[str]
        text      : str
    """
    # per-generation seed so runs are reproducible
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    inputs     = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    cur_ids    = inputs["input_ids"].clone()
    step_reps  = []
    tok_ids    = []
    tok_strs   = []

    for _ in range(cfg.MAX_TOKENS):
        with torch.inference_mode():
            out = mdl(cur_ids, output_hidden_states=True)
        hs = torch.stack(out.hidden_states)            # (L+1, B, S, H)
        step_reps.append(hs[:, 0, -1, :].cpu().float()) # (L+1, H)

        logits = out.logits[0, -1, :]
        if cfg.TEMPERATURE <= 0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            probs    = torch.softmax(logits / cfg.TEMPERATURE, dim=-1)
            next_tok = torch.multinomial(probs, 1)

        tid = next_tok.item()
        tok_ids.append(tid)
        tok_strs.append(tok.decode([tid]))
        cur_ids = torch.cat([cur_ids, next_tok.unsqueeze(0)], dim=1)
        if tid == tok.eos_token_id:
            break

    text = tok.decode(tok_ids, skip_special_tokens=True)
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
    """cos(δ_i, δ_{i+1}) across layers."""
    deltas = [reps[i+1] - reps[i] for i in range(n-1)]
    return np.array([
        cos(deltas[i], deltas[i+1]) for i in range(len(deltas)-1)
    ])

def sign_flip(reps, n):
    """Fraction of dims that flip delta sign across consecutive layers."""
    deltas = [(reps[i+1] - reps[i]).numpy() for i in range(n-1)]
    ratios = []
    for i in range(len(deltas)-1):
        sa, sb   = np.sign(deltas[i]), np.sign(deltas[i+1])
        nz       = (sa != 0) & (sb != 0)
        ratios.append(0.5 if nz.sum() == 0
                      else (sa[nz] != sb[nz]).sum() / nz.sum())
    return np.array(ratios)


# ── Encoding analysis ─────────────────────────────────────────────────────────
def run_encoding(mdl, tok, pair, n_layers):
    hs_b, toks_b, ids_b = get_hidden_states(mdl, tok, pair["benign"])
    hs_h, toks_h, ids_h = get_hidden_states(mdl, tok, pair["harmful"])

    lr = n_layers + 1 - cfg.EXCLUDE_LAST_N   # usable layer range

    # resolve positions
    if cfg.TOKEN_POSITIONS is not None:
        seq_len = min(hs_b.shape[1], hs_h.shape[1])
        positions = [seq_len + p if p < 0 else p for p in cfg.TOKEN_POSITIONS]
    else:
        # auto: divergent positions + last
        mn = min(len(ids_b), len(ids_h))
        div = [i for i in range(mn) if ids_b[i] != ids_h[i]]
        positions = sorted(set(div + [mn - 1]))

    n_pos = len(positions)
    L     = lr

    # pre-allocate
    out = {
        "enc_positions":  np.array(positions),
        "enc_toks_b":     np.array(toks_b, dtype=object),
        "enc_toks_h":     np.array(toks_h, dtype=object),
        "enc_l2":         np.zeros((n_pos, L)),
        "enc_cos":        np.zeros((n_pos, L)),
        "enc_l2_vel":     np.zeros((n_pos, L)),
        "enc_turb_b":     np.zeros((n_pos, L-2)),
        "enc_turb_h":     np.zeros((n_pos, L-2)),
        "enc_flip_b":     np.zeros((n_pos, L-2)),
        "enc_flip_h":     np.zeros((n_pos, L-2)),
    }

    for i, pos in enumerate(positions):
        rb = hs_b[:lr, pos, :]
        rh = hs_h[:lr, pos, :]
        l2v = l2_per_layer(rb, rh, lr)
        out["enc_l2"][i]     = l2v
        out["enc_cos"][i]    = cos_per_layer(rb, rh, lr)
        out["enc_l2_vel"][i] = np.gradient(l2v)
        out["enc_turb_b"][i] = turbulence(rb, lr)
        out["enc_turb_h"][i] = turbulence(rh, lr)
        out["enc_flip_b"][i] = sign_flip(rb, lr)
        out["enc_flip_h"][i] = sign_flip(rh, lr)

    return out


# ── Generation analysis ───────────────────────────────────────────────────────
def run_generation(mdl, tok, pair, n_layers):
    al = n_layers // 2   # analysis layer

    reps_b, tstrs_b, text_b = generate(mdl, tok, pair["benign"])
    reps_h, tstrs_h, text_h = generate(mdl, tok, pair["harmful"])

    T = min(len(reps_b), len(reps_h))

    l2v   = np.array([l2(reps_b[t][al],  reps_h[t][al])  for t in range(T)])
    cosv  = np.array([cos(reps_b[t][al], reps_h[t][al])  for t in range(T)])

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
        "gen_l2":       l2v,
        "gen_cos":      cosv,
        "gen_dir_stab": dir_stab,
        "gen_turb_b":   np.array(dturb_b),
        "gen_turb_h":   np.array(dturb_h),
        "gen_toks_b":   np.array(tstrs_b[:T], dtype=object),
        "gen_toks_h":   np.array(tstrs_h[:T], dtype=object),
        "gen_text_b":   np.array(text_b),
        "gen_text_h":   np.array(text_h),
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
                print(f"  [{pair['name']}] exists, skipping. Use --rerun to overwrite.")
                continue

            print(f"  Running pair: {pair['name']}")
            enc = run_encoding(mdl, tok, pair, n_layers)
            gen = run_generation(mdl, tok, pair, n_layers)

            meta = {
                "model":      np.array(mname),
                "pair":       np.array(pair["name"]),
                "benign":     np.array(pair["benign"]),
                "harmful":    np.array(pair["harmful"]),
                "n_layers":   np.array(n_layers),
                "seed":       np.array(cfg.SEED),
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