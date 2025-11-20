# per_layer_readout_suite.py
# Measures where arithmetic becomes linearly decodable in GPT-2:
#   • logit-diff at resid_pre / resid_mid (after attn) / resid_post (after mlp) / final ln
#   • finite-difference gain bars per sublayer
#   • carry vs no-carry splits
#
# Usage (headless/Vast):
#   python per_layer_readout_suite.py --n 160 --model gpt2 --seed 17 --out runs/readout_suite
#
# Outputs:
#   runs/readout_suite/
#     readout_curve_mean.png                (overall mean with std band)
#     readout_curve_mean_carry.png          (carry only)
#     readout_curve_mean_nocarry.png        (no-carry only)
#     gain_bars.png                         (where evidence is added)
#     stage_means.csv                       (per layer/stage means & stds)
#     per_example_diffs.csv                 (all examples, all stages)
#     manifest.json

import os, json, random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer

# -------------------- helpers --------------------

def set_seed(s):
    import numpy as np, torch, random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def gen_add_pairs(n=200, max_val=99):
    """Return list of (prompt, a, b, ans_str, foil_str, carry_flag) for +.
       We create a close foil (±1 or ±2) but later we score answer vs foil logits.”
    """
    pairs = []
    for _ in range(n*3):  # oversample to allow filtering
        a = random.randint(0, max_val)
        b = random.randint(0, max_val)
        ans = a + b
        # choose a near foil within [0, 198]
        delta = random.choice([-2,-1,1,2])
        foil = min(198, max(0, ans + delta))
        carry = (a % 10 + b % 10) >= 10
        prompt = f"Q: {a} + {b} = ? A:"
        pairs.append((prompt, a, b, str(ans), str(foil), carry))
    return pairs

def pick_single_token_id(model, s):
    """Prefer leading space version; fall back to no space. Return token id or None."""
    for cand in (f" {s}", s):
        tid = model.to_single_token(cand)
        if tid is not None:
            return tid
    return None

def collect_diffs(model, pairs, max_keep):
    """Keep only examples where BOTH answer and foil are single tokens.
       Return list of dict rows with per-stage logit diffs per layer."""
    device = model.cfg.device
    rows = []

    keep = 0
    for (prompt, a, b, ans, foil, carry) in pairs:
        if keep >= max_keep: break

        ans_id = pick_single_token_id(model, ans)
        foil_id = pick_single_token_id(model, foil)
        if ans_id is None or foil_id is None:
            continue  # skip multi-token answers/foils for this analysis

        toks = model.to_tokens(prompt, prepend_bos=True).to(device)
        _, cache = model.run_with_cache(toks)

        # Stages:
        #   resid_pre[l]                    # before attention
        #   resid_mid[l]  = resid_pre + attn_out
        #   resid_post[l] = resid_mid + mlp_out
        #   final_ln      = ln_final(resid_post[last])
        diffs_pre   = []
        diffs_mid   = []
        diffs_post  = []

        for l in range(model.cfg.n_layers):
            resid_pre = cache["resid_pre", l][:, -1, :]   # [1, d_model]
            resid_mid = cache["resid_mid", l][:, -1, :]
            resid_post= cache["resid_post", l][:, -1, :]

            logits_pre  = model.unembed(resid_pre)[0]
            logits_mid  = model.unembed(resid_mid)[0]
            logits_post = model.unembed(resid_post)[0]

            diffs_pre.append(  (logits_pre[ans_id]  - logits_pre[foil_id]).item() )
            diffs_mid.append(  (logits_mid[ans_id]  - logits_mid[foil_id]).item() )
            diffs_post.append( (logits_post[ans_id] - logits_post[foil_id]).item() )

        # final layernorm output (true readout)
        resid_last = cache["resid_post", model.cfg.n_layers - 1]
        ln_final = model.ln_final(resid_last)[:, -1, :]
        logits_final = model.unembed(ln_final)[0]
        diff_final = (logits_final[ans_id] - logits_final[foil_id]).item()

        # store rows per layer/stage for this example
        for l in range(model.cfg.n_layers):
            rows.append({
                "prompt": prompt,
                "a": a, "b": b, "ans": int(ans), "foil": int(foil),
                "carry": int((a % 10 + b % 10) >= 10),
                "layer": l,
                "stage": "resid_pre",
                "logit_diff": diffs_pre[l],
            })
            rows.append({
                "prompt": prompt,
                "a": a, "b": b, "ans": int(ans), "foil": int(foil),
                "carry": int((a % 10 + b % 10) >= 10),
                "layer": l,
                "stage": "resid_mid",
                "logit_diff": diffs_mid[l],
            })
            rows.append({
                "prompt": prompt,
                "a": a, "b": b, "ans": int(ans), "foil": int(foil),
                "carry": int((a % 10 + b % 10) >= 10),
                "layer": l,
                "stage": "resid_post",
                "logit_diff": diffs_post[l],
            })
        rows.append({
            "prompt": prompt,
            "a": a, "b": b, "ans": int(ans), "foil": int(foil),
            "carry": int((a % 10 + b % 10) >= 10),
            "layer": model.cfg.n_layers,
            "stage": "final_ln",
            "logit_diff": diff_final,
        })

        keep += 1

    return rows

def plot_readout_curves(df, outdir, title_suffix=""):
    # mean ± std across prompts for each stage
    stages_order = ["resid_pre", "resid_mid", "resid_post", "final_ln"]
    colors = {"resid_pre":None, "resid_mid":None, "resid_post":None, "final_ln":None}  # let MPL pick

    plt.figure(figsize=(9,5))
    for st in stages_order:
        sub = df[df.stage == st]
        g = sub.groupby("layer")["logit_diff"]
        means = g.mean()
        stds  = g.std().fillna(0.0)
        layers = means.index.values
        vals   = means.values
        plt.plot(layers, vals, marker="o", label=st)
        # error band for non-final stage
        if st != "final_ln":
            plt.fill_between(layers, vals-stds.values, vals+stds.values, alpha=0.15)
    plt.title(f"Logit difference vs layer {title_suffix}".strip())
    plt.xlabel("Layer")
    plt.ylabel("Mean logit(correct) − logit(foil)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"readout_curve_mean{title_suffix.replace(' ','_')}.png", dpi=160)

def plot_gain_bars(df, outdir):
    # Finite differences: pre→mid (attn gain), mid→post (mlp gain), post(l)→pre(l+1) (residual carry)
    # Compute means across prompts (ignoring carry split for this figure).
    g = df.pivot_table(index=["prompt","layer"], columns="stage", values="logit_diff")
    # Filter rows where all three stages exist
    g = g.dropna(subset=["resid_pre","resid_mid","resid_post"], how="any")

    attn_gain = (g["resid_mid"] - g["resid_pre"]).groupby("layer").mean()
    mlp_gain  = (g["resid_post"] - g["resid_mid"]).groupby("layer").mean()

    # carry-over to next layer: resid_pre(l+1) - resid_post(l)
    # Need aligned indices:
    post = g["resid_post"].groupby("layer").mean()
    pre_next = g["resid_pre"].groupby("layer").mean().shift(-1)
    carry_gain = (pre_next - post)[:-1]  # drop last NaN

    L = sorted(attn_gain.index.tolist())
    plt.figure(figsize=(10,5))
    width = 0.35
    plt.bar([l-0.15 for l in L], attn_gain.values, width=0.3, label="attn gain (pre→mid)")
    plt.bar([l+0.15 for l in L], mlp_gain.values,  width=0.3, label="mlp gain (mid→post)")
    # Make a light line for carry_gain
    plt.plot(carry_gain.index, carry_gain.values, marker="o", linestyle="--", label="carry to next (post→pre+1)")
    plt.title("Where evidence is added (finite differences)")
    plt.xlabel("Layer")
    plt.ylabel("Δ logit-diff")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "gain_bars.png", dpi=160)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n", type=int, default=160, help="target number of kept examples")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="runs/readout_suite")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device)
    print(f"Loaded {args.model} on {device}")

    pairs = gen_add_pairs(n=args.n*2)       # oversample to allow filtering
    rows  = collect_diffs(model, pairs, max_keep=args.n)

    if len(rows) == 0:
        raise RuntimeError("No single-token answer+foil examples found. Consider lowering max_val or n.")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_example_diffs.csv", index=False)

    # Overall plots
    plot_readout_curves(df, outdir, title_suffix="")
    # carry split
    plot_readout_curves(df[df.carry == 1], outdir, title_suffix=" (carry)")
    plot_readout_curves(df[df.carry == 0], outdir, title_suffix=" (nocarry)")

    # Gain bars
    plot_gain_bars(df, outdir)

    # Stage means table
    stage_means = df.groupby(["layer","stage"])["logit_diff"].agg(["mean","std","count"]).reset_index()
    stage_means.to_csv(outdir / "stage_means.csv", index=False)

    # Manifest
    (outdir / "manifest.json").write_text(json.dumps({
        "model": args.model,
        "n_kept_examples": int(df["prompt"].nunique()),
        "seed": args.seed,
        "outdir": str(outdir),
        "notes": "logit-diff at resid_pre/mid/post and final ln; carry split; finite-diff gains."
    }, indent=2))

    print("Saved outputs to", outdir)

if __name__ == "__main__":
    main()

