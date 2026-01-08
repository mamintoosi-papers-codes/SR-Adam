import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

def parse_list_arg(raw, value_type=str):
    if raw is None:
        return None
    raw = str(raw)
    if raw.upper() == "ALL":
        return None
    for sep in ["|", ";", ","]:
        if sep in raw:
            tokens = [t.strip() for t in raw.split(sep) if t.strip()]
            break
    else:
        tokens = [raw.strip()]
    if value_type == float:
        return [float(t) for t in tokens]
    elif value_type == int:
        return [int(t) for t in tokens]
    else:
        return tokens

def load_runs(base_dir):
    files = sorted(base_dir.glob("run_*_metrics.csv"))
    runs = {}
    for rf in files:
        try:
            df = pd.read_csv(rf)
        except Exception:
            continue
        # run index from filename
        name = rf.stem  # run_1_metrics
        parts = name.split("_")
        run_id = parts[1] if len(parts) > 1 else str(len(runs) + 1)
        runs[run_id] = df
    return runs

def extract_metric(df, mode="best"):
    if "Test Acc" not in df.columns:
        return None
    if mode == "final":
        return float(df["Test Acc"].iloc[-1])
    else:
        return float(df["Test Acc"].max())

def main():
    parser = argparse.ArgumentParser(description="Paired stat tests: Adam vs SR-Adam")
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--noise", type=str, default="ALL")
    parser.add_argument("--batch_size", type=str, default="512")
    parser.add_argument("--model", type=str, default="simplecnn")
    parser.add_argument("--mode", type=str, choices=["final", "best"], default="best")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--output_tex", type=str, default=None)
    args = parser.parse_args()

    datasets = parse_list_arg(args.dataset) or ["CIFAR10", "CIFAR100"]
    noises = parse_list_arg(args.noise, float) or [0.0, 0.05, 0.1]
    batch_sizes = parse_list_arg(args.batch_size, int) or [512]

    results = []
    for ds in datasets:
        for nz in noises:
            base = Path("results") / ds / args.model / f"noise_{nz}"
            adam_dir = base / "Adam"
            sradam_dir = base / "SR-Adam"
            for bs in batch_sizes:
                a_bs = adam_dir / f"batch_size_{bs}"
                s_bs = sradam_dir / f"batch_size_{bs}"
                if not a_bs.exists() or not s_bs.exists():
                    continue
                adam_runs = load_runs(a_bs)
                sradam_runs = load_runs(s_bs)
                # align run ids intersection
                common_ids = sorted(set(adam_runs.keys()) & set(sradam_runs.keys()))
                if not common_ids:
                    continue
                a_vals, s_vals = [], []
                for rid in common_ids:
                    a_val = extract_metric(adam_runs[rid], args.mode)
                    s_val = extract_metric(sradam_runs[rid], args.mode)
                    if a_val is None or s_val is None:
                        continue
                    a_vals.append(a_val)
                    s_vals.append(s_val)
                if not a_vals or not s_vals:
                    continue
                a_arr = np.array(a_vals)
                s_arr = np.array(s_vals)
                # paired tests
                t_stat, t_p = ttest_rel(a_arr, s_arr, nan_policy="omit")
                try:
                    w_stat, w_p = wilcoxon(a_arr, s_arr)
                except ValueError:
                    w_stat, w_p = (np.nan, np.nan)
                results.append({
                    "dataset": ds, "noise": nz, "batch_size": bs,
                    "n_pairs": len(a_arr),
                    "adam_mean": float(np.mean(a_arr)),
                    "sradam_mean": float(np.mean(s_arr)),
                    "ttest_stat": float(t_stat), "ttest_p": float(t_p), "ttest_sig": bool(t_p < args.alpha),
                    "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p), "wilcoxon_sig": bool(w_p < args.alpha)
                })

    if not results:
        print("No paired runs found for Adam vs SR-Adam.")
        return

    df = pd.DataFrame(results).sort_values(["dataset", "noise", "batch_size"])
    print(df.to_string(index=False))

    # outputs
    ds_suf = "_ds" + "-".join(str(ds) for ds in sorted(datasets))
    bs_suf = "_bs" + "-".join(str(bs) for bs in sorted(batch_sizes))
    if args.output_csv is None:
        args.output_csv = f"paper/stat_tests_adam_vs_sradam{ds_suf}{bs_suf}.csv"
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved CSV: {args.output_csv}")

    if args.output_tex is None:
        args.output_tex = f"paper/stat_tests_adam_vs_sradam_{ds_suf}{bs_suf}.tex"
    lines = ["% Auto-generated Adam vs SR-Adam stat tests\n", "\\usepackage{booktabs}\n\n"]
    for ds in datasets:
        lines.append(f"% {ds}\n")
        lines.append("\\begin{table}[t]\n\\centering\n\\begin{tabular}{lcccccc}\n\\toprule\n")
        lines.append("Noise & Batch & n & Adam Mean & SR-Adam Mean & t p-val & Wilcoxon p-val \\\\\n\\midrule\n")
        sub = df[df.dataset == ds]
        for _, r in sub.iterrows():
            lines.append(f"{r.noise} & {int(r.batch_size)} & {int(r.n_pairs)} & {r.adam_mean:.2f} & {r.sradam_mean:.2f} & {r.ttest_p:.3g} & {r.wilcoxon_p:.3g} \\\\\n")
        lines.append("\\bottomrule\n\\end{tabular}\n")
        # Avoid f-strings to keep LaTeX braces literal
        lines.append("\\caption{Paired tests (mode=%s) for Adam vs SR-Adam on %s}\n" % (args.mode, ds))
        lines.append("\\label{tab:stat_adam_sradam_%s}\n\\end{table}\n\n" % (ds.lower()))
    with open(args.output_tex, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Saved LaTeX: {args.output_tex}")

if __name__ == "__main__":
    main()