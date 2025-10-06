import random, json, argparse
random.seed(17)

def gen_pair(nd):
    a = random.randint(10**(nd-1), 10**nd - 1)
    b = random.randint(10**(nd-1), 10**nd - 1)
    return a, b, a + b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--min_digits", type=int, default=3)
    ap.add_argument("--max_digits", type=int, default=8)
    ap.add_argument("--out", type=str, default="data/addition_train.jsonl")
    args = ap.parse_args()

    with open(args.out, "w") as f:
        for _ in range(args.n):
            nd = random.randint(args.min_digits, args.max_digits)
            a,b,c = gen_pair(nd)
            prompt = f"{a} + {b} ="
            target = str(c)
            f.write(json.dumps({"prompt": prompt, "target": target}) + "\n")

if __name__ == "__main__":
    main()

