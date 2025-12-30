import os
import pandas as pd

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_CSV = os.path.join(RESULTS_ROOT, "summary_statistics.csv")
TABLES_TEX = os.path.join(os.path.dirname(__file__), "tables.tex")

NOISE_LEVELS = ["0.0", "0.05", "0.1"]
OPTIMIZERS = ["Adam", "SR-Adam"]


def _load_summary():
    df = pd.read_csv(SUMMARY_CSV, index_col=0)
    df = df.reset_index().rename(columns={"index": "key"})
    parts = df["key"].str.split("|", expand=True)
    df["dataset"] = parts[0]
    df["noise"] = parts[1].str.replace("noise_", "", regex=False)
    df["optimizer"] = parts[2]
    return df


def _format_row(mean, std, is_best):
    cell = f"{mean:.2f} \\pm {std:.2f}"
    return f"\\textbf{{{cell}}}" if is_best else cell


def build_table(df, dataset, mean_col, std_col, caption, label):
    subset = df[(df["dataset"] == dataset) & (df["optimizer"].isin(OPTIMIZERS)) & (df["noise"].isin(NOISE_LEVELS))]

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\begin{tabular}{lcc}")
    lines.append("    \\toprule")
    lines.append("    Noise & Adam & SR-Adam \\\\")
    lines.append("    \\midrule")

    for noise in NOISE_LEVELS:
        row = subset[subset["noise"] == noise].set_index("optimizer")
        vals = {}
        for opt in OPTIMIZERS:
            if opt not in row.index:
                vals[opt] = (float("nan"), float("nan"))
            else:
                vals[opt] = (row.loc[opt, mean_col], row.loc[opt, std_col])
        best_opt = max(OPTIMIZERS, key=lambda o: vals[o][0])
        adam_cell = _format_row(*vals["Adam"], is_best=(best_opt == "Adam"))
        sr_cell = _format_row(*vals["SR-Adam"], is_best=(best_opt == "SR-Adam"))
        lines.append(f"    {noise} & {adam_cell} & {sr_cell} \\")

    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def main():
    df = _load_summary()

    table1 = []
    table2 = []
    for dataset in ["CIFAR10", "CIFAR100"]:
        table1.append(
            build_table(
                df,
                dataset,
                mean_col="final_mean",
                std_col="final_std",
                caption=f"Final test accuracy (mean $\\pm$ std) on {dataset}.",
                label=f"tab:{dataset.lower()}-final",
            )
        )
        table2.append(
            build_table(
                df,
                dataset,
                mean_col="best_mean",
                std_col="best_std",
                caption=f"Best test accuracy over epochs (mean $\\pm$ std) on {dataset}.",
                label=f"tab:{dataset.lower()}-best",
            )
        )

    content = [
        "% Auto-generated; do not edit by hand",
        "\\usepackage{booktabs}",
        "",
        "% Table 1: Final accuracy",
        *table1,
        "% Table 2: Best accuracy",
        *table2,
    ]

    with open(TABLES_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
    print(f"Wrote LaTeX tables to {TABLES_TEX}")


if __name__ == "__main__":
    main()
