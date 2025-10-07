
"""
mech_math_morning.py
Quick CLI to run two fast experiments for mechanistic interpretability on math in LLMs.

Experiments
----------
A: Logit-Diff Probe on synthetic 1-2 digit arithmetic templates.
   - Measures logit(correct) - logit(foil) across controlled corruptions.
   - Saves CSV + optional plot.

B: Path Patching (residual stream) around carry/operand token positions.
   - Identifies layers/positions most causally important.
   - Saves heatmap CSV and optional plot.

Requirements
------------
pip install transformer-lens torch matplotlib pandas numpy
(Use the model 'gpt2' or other small HF model supported by TransformerLens.)

Usage
-----
python mech_math_morning.py --exp A --model gpt2 --n 128 --seed 17
python mech_math_morning.py --exp B --model gpt2 --n 64 --seed 17 --plot
"""
import argparse, os, json, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    from transformer_lens import HookedTransformer, utils as tl_utils
except Exception as e:
    print("TransformerLens import failed. Please install: pip install transformer-lens")
    raise

# ---------------- Utils ----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def gen_add_sub_prompts(n=128, max_val=99, op="+"):
    # Returns list of (prompt, answer_str, foil_str)
    data = []
    for _ in range(n):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        if op == "+":
            ans = a + b
            foil = ans + random.choice([-1, 1, 2, -2])  # simple foil
            prompt = f"Q: {a} + {b} = ? A:"
        else:
            ans = a - b
            foil = ans + random.choice([-1, 1, 2, -2])
            prompt = f"Q: {a} - {b} = ? A:"
        data.append((prompt, str(ans), str(foil)))
    return data

def tokenize_prompts(model, prompts):
    toks = model.to_tokens(prompts, prepend_bos=True)
    return toks

def logit_for_token(model, logits, token_str):
    tok_id = model.to_single_token(token_str)
    if tok_id is None:
        # fallback: take first char
        tok_id = model.to_single_token(token_str[0])
    return logits[..., tok_id]

# ------------- Experiment A: Logit-Diff Probe -------------
def run_exp_A(model_name: str, n: int, seed: int, outdir: Path, plot: bool):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    data = gen_add_sub_prompts(n=n, max_val=99, op="+")
    prompts = [p for (p, a, f) in data]

    toks = tokenize_prompts(model, prompts).to(device)
    with torch.no_grad():
        logits, cache = model.run_with_cache(toks)

    # We measure logit-diff on the final position
    final_logits = logits[:, -1, :]
    rows = []
    for i, (prompt, ans, foil) in enumerate(data):
        ld = (logit_for_token(model, final_logits[i], ans) -
              logit_for_token(model, final_logits[i], foil)).item()
        rows.append({"i": i, "prompt": prompt, "answer": ans, "foil": foil, "logit_diff": float(ld)})
    df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "expA_logitdiff.csv", index=False)

    if plot:
        plt.figure()
        plt.hist(df["logit_diff"], bins=30)
        plt.title("Experiment A: Logit Difference Distribution")
        plt.xlabel("logit(answer) - logit(foil)")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(outdir / "expA_logitdiff_hist.png")

    # Quick summary
    summary = {
        "mean_logit_diff": float(df["logit_diff"].mean()),
        "median_logit_diff": float(df["logit_diff"].median()),
        "p_pos": float((df["logit_diff"] > 0).mean()),
        "n": int(n),
        "model": model_name,
    }
    (outdir / "expA_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved Experiment A outputs to", outdir)

# ------------- Experiment B: Path Patching (residual stream) -------------
# Minimal, fast variant: patching layer outputs from clean to corrupted prompts
# Corruption: swap operands (b, a) while keeping surface template
def make_clean_corrupted_pairs(n=64, max_val=99):
    pairs = []
    for _ in range(n):
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        clean = f"Q: {a} + {b} = ? A:"
        corrupted = f"Q: {b} + {a} = ? A:"  # foil: swapped operands
        ans = str(a + b)
        foil = str(b + a)  # same numerically, but we will score other foils later if needed
        pairs.append((clean, corrupted, ans, foil))
    return pairs

def residual_stream_patching(model, toks_clean, toks_corr, layer_idx, device):
    # Run once to get clean cache
    _, cache_clean = model.run_with_cache(toks_clean, remove_batch_dim=False)
    # Hook function to replace residual at layer output
    def hook_fn(resid_post, hook):
        # resid_post: [B, pos, d_model]
        resid_post[:] = cache_clean[hook.name]
        return resid_post

    # Apply hook on corrupted pass
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        logits_patched = model(toks_corr)
    return logits_patched

def run_exp_B(model_name: str, n: int, seed: int, outdir: Path, plot: bool):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(model_name, device=device)

    pairs = make_clean_corrupted_pairs(n=n)
    clean_prompts = [c for (c, _, _, _) in pairs]
    corr_prompts  = [k for (_, k, _, _) in pairs]
    ans_tokens = [a for (_, _, a, _) in pairs]

    toks_clean = model.to_tokens(clean_prompts, prepend_bos=True).to(device)
    toks_corr  = model.to_tokens(corr_prompts, prepend_bos=True).to(device)

    # Baseline (uncorrupted vs corrupted) at final position
    with torch.no_grad():
        logits_clean = model(toks_clean)
        logits_corr  = model(toks_corr)

    def get_ld_row(logits_batch, answers):
        final_logits = logits_batch[:, -1, :]
        out = []
        for i, ans in enumerate(answers):
            ld = (final_logits[i, model.to_single_token(ans)] -
                  final_logits[i, model.to_single_token(str(int(ans)+1))]).item()
            out.append(ld)
        return np.array(out)

    baseline_clean = get_ld_row(logits_clean, ans_tokens)
    baseline_corr  = get_ld_row(logits_corr,  ans_tokens)

    L = model.cfg.n_layers
    impacts = []
    for layer_idx in range(L):
        logits_patched = residual_stream_patching(model, toks_clean, toks_corr, layer_idx, device)
        patched_scores = get_ld_row(logits_patched, ans_tokens)
        # Impact: recovery from corrupted towards clean
        impact = (patched_scores - baseline_corr) / (baseline_clean - baseline_corr + 1e-6)
        impacts.append(impact.mean())

    df = pd.DataFrame({"layer": list(range(L)), "impact_recovery": impacts})
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "expB_layer_impacts.csv", index=False)

    if plot:
        plt.figure()
        plt.plot(df["layer"], df["impact_recovery"], marker="o")
        plt.title("Experiment B: Residual Stream Patching Impact by Layer")
        plt.xlabel("Layer")
        plt.ylabel("Recovery fraction (0=none, 1=clean)")
        plt.tight_layout()
        plt.savefig(outdir / "expB_layer_impacts.png")

    print("Saved Experiment B outputs to", outdir)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B"], required=True, help="Which experiment to run")
    ap.add_argument("--model", default="gpt2", help="TransformerLens model name")
    ap.add_argument("--n", type=int, default=128, help="Number of samples")
    ap.add_argument("--seed", type=int, default=17, help="PRNG seed")
    ap.add_argument("--out", default=None, help="Output directory (default: runs/morning_<date>/<exp>)")
    ap.add_argument("--plot", action="store_true", help="Save a simple plot")
    args = ap.parse_args()

    date_tag = os.environ.get("DATE_TAG")
    if not date_tag:
        from datetime import datetime
        date_tag = datetime.now().strftime("%Y%m%d")
    base_out = Path(args.out) if args.out else Path(f"runs/morning_{date_tag}/{args.exp}")
    base_out.mkdir(parents=True, exist_ok=True)

    if args.exp == "A":
        run_exp_A(args.model, args.n, args.seed, base_out, args.plot)
    else:
        run_exp_B(args.model, min(args.n, 256), args.seed, base_out, args.plot)

    # Write a small manifest
    (base_out / "manifest.json").write_text(json.dumps({
        "exp": args.exp,
        "model": args.model,
        "n": args.n,
        "seed": args.seed,
        "plot": args.plot,
        "outdir": str(base_out)
    }, indent=2))

if __name__ == "__main__":
    main()
