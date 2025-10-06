import json, argparse, torch
from transformer_lens import HookedTransformer

def batchify(lst, bs): 
    for i in range(0, len(lst), bs):
        yield lst[i:i+bs]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/addition_train.jsonl")
    ap.add_argument("--model", default="gpt2")          # later: pythia-160m
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_eval", type=int, default=5000)
    args = ap.parse_args()

    model = HookedTransformer.from_pretrained(args.model, device=args.device)
    samples = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.max_eval: break
            samples.append(json.loads(line))

    correct = 0; total = 0
    for chunk in batchify(samples, 64):
        prompts = [s["prompt"] for s in chunk]
        targets = [s["target"] for s in chunk]
        toks = model.to_tokens(prompts, prepend_bos=True).to(args.device)
        with torch.no_grad():
            logits = model(toks)  # [B, T, V]
        # greedy decode next few tokens to length of target
        outs = model.to_string(logits[:, -1:, :].argmax(-1))
        # coarse: just compare first digit; we’ll refine soon
        for t, g in zip(outs, targets):
            correct += int(t.strip()[0] == g[0])
            total += 1

    print(f"First-digit accuracy: {correct/total:.3f} on {total} samples")

if __name__ == "__main__":
    main()

