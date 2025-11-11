import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import numpy as np

# --- Load GPT-2 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)

# --- Prompts ---
prompts = [
    ("12 + 7 =", "19", "18"),
    ("23 + 8 =", "31", "30"),
    ("45 + 6 =", "51", "52"),
    ("99 + 1 =", "100", "101"),
]

# --- Helper ---
def to_tok(model, s): return model.to_tokens(s, prepend_bos=True).to(device)

def logit_diff_at_layer(model, prompt, ans, foil):
    toks = model.to_tokens(prompt, prepend_bos=True).to(model.cfg.device)
    ans_id = model.to_single_token(" " + ans) or model.to_single_token(ans)
    foil_id = model.to_single_token(" " + foil) or model.to_single_token(foil)
    if ans_id is None or foil_id is None:
        return None

    _, cache = model.run_with_cache(toks)

    diffs = []

    # (A) BEFORE any block: resid_pre at layer 0
    resid0 = cache["resid_pre", 0][:, -1, :]            # [1, d_model]
    logits0 = model.unembed(resid0)                     # [1, vocab]
    diffs.append(( -1, (logits0[0, ans_id] - logits0[0, foil_id]).item() ))

    # (B) AFTER each block l = 0..n_layers-1: resid_post
    for l in range(model.cfg.n_layers):
        resid_post = cache["resid_post", l][:, -1, :]
        logits_l = model.unembed(resid_post)
        diffs.append(( l, (logits_l[0, ans_id] - logits_l[0, foil_id]).item()))

    # (C) FINAL readout: after ln_final (this is what the model actually uses)
    resid_final = model.ln_final(cache["resid_post", model.cfg.n_layers - 1])[:, -1, :]
    logits_final = model.unembed(resid_final)
    diffs.append(( model.cfg.n_layers, (logits_final[0, ans_id] - logits_final[0, foil_id]).item() ))

    return diffs  # list of (layer_index, diff); layer_index=-1 = pre-block, n_layers = final LN


def logit_diff_at_layer_wrong(model, prompt, ans, foil):
    toks = to_tok(model, prompt)
    ans_id = model.to_single_token(" " + ans) or model.to_single_token(ans)
    foil_id = model.to_single_token(" " + foil) or model.to_single_token(foil)
    if ans_id is None or foil_id is None:
        return None

    # Run with cache
    _, cache = model.run_with_cache(toks)
    print(cache.keys())
    diffs = []
    for layer in range(model.cfg.n_layers + 1):  # include layer 0 (embed)
        resid = cache["resid_post", layer]  # [1, seq, d_model]
        last_tok = resid[:, -1, :]          # final position
        logits = model.unembed(last_tok)    # [1, vocab]
        diff = (logits[0, ans_id] - logits[0, foil_id]).item()
        diffs.append(diff)
    return diffs

# --- Collect trajectories ---
all_diffs = []
for p, a, f in prompts:
    diffs = logit_diff_at_layer(model, p, a, f)
    all_diffs.append(diffs)

# Average across prompts
mean_diffs = np.mean(np.array(all_diffs), axis=0)

# --- Plot ---
print(mean_diffs)
plt.figure(figsize=(8,4))
plt.plot(range(len(mean_diffs)), mean_diffs, marker="o")
plt.title("Logit Difference (Correct - Foil) vs Layer — GPT-2")
plt.xlabel("Layer index")
plt.ylabel("Mean logit difference")
plt.grid(True)
plt.show()

plt.savefig("per_layer_logit_diff.png", bbox_inches="tight")
print("Saved figure to per_layer_logit_diff.png")

