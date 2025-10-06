import torch, json, argparse
import re
from transformer_lens import HookedTransformer
from transformer_lens import utils

def make_pairs(ex):
    # ex["prompt"] looks like "12345 + 67890 ="
    p = ex["prompt"]

    # Keep only the left-hand side "A + B"
    lhs = p.split("=")[0].strip()            # "12345 + 67890"
    a_str, b_str = [s.strip() for s in lhs.split("+")]

    # Ensure b_str ends with digits (be robust to spaces)
    m = re.search(r"(\d+)\s*$", b_str)
    if not m:
        raise ValueError(f"Could not parse second addend from: {b_str!r}")
    b_digits = list(m.group(1))
    # Perturb the last digit (mod 10) to create a minimally corrupted input
    b_digits[-1] = str((int(b_digits[-1]) + 1) % 10)
    b_pert = b_str[:m.start(1)] + "".join(b_digits)

    corrupt = f"{a_str} + {b_pert} ="       # keep exact formatting
    clean = p                               # original prompt
    return clean, corrupt

'''
def make_pairs(ex):
    # clean: "A + B =" ; corrupt: swap last digit of B
    p = ex["prompt"]; tgt = ex["target"]
    a,b = p.split(" + ")
    b = b[:-1] + str((int(b[-1]) + 1) % 10)  # cheap perturb
    corrupt = a + " + " + b
    return p, corrupt
'''

def metric(model, toks, gold_first_digit):
    with torch.no_grad():
        logits = model(toks)
    pred = model.to_string(logits[:, -1:, :].argmax(-1))[0].strip()[:1]
    return float(pred == gold_first_digit)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/addition_train.jsonl")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    with open(args.data) as f:
        ex = json.loads(next(f))
    clean, corrupt = make_pairs(ex)
    gold = ex["target"][0]

    toks_clean = model.to_tokens(clean, prepend_bos=True).to(args.device)
    toks_corrupt = model.to_tokens(corrupt, prepend_bos=True).to(args.device)

    base_clean = metric(model, toks_clean, gold)
    base_corrupt = metric(model, toks_corrupt, gold)
    print("clean ok?", base_clean, "corrupt ok?", base_corrupt)

    # simple per-layer patch: copy resid_post from clean into corrupt, scan layers
    L = model.cfg.n_layers
    effects = []
    for layer in range(L):
        def hook_patch(resid_post, hook):
            # borrow resid_post at this layer from clean pass
            with torch.no_grad():
                clean_cache = {}
                model.run_with_cache(toks_clean, names_filter=lambda n: f"blocks.{layer}.hook_resid_post" in n, cache_dict=clean_cache)
                return clean_cache[f'blocks.{layer}.hook_resid_post'].clone()

        score = model.run_with_hooks(
            toks_corrupt,
            fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_patch)],
            return_type=None
        )
        # re-run metric after patched forward (logits cached on model)
        patched_acc = metric(model, toks_corrupt, gold)
        effects.append((layer, patched_acc))
        # clear hooks state just in case
        model.reset_hooks()

    print("layer→patched_acc:", effects)

if __name__ == "__main__":
    main()

