"""
Generate LaTeX table with SimpleCNN architecture and parameter count.
"""


def count_parameters_simplecnn():
    """Calculate SimpleCNN total parameters."""
    # Conv1: 3 input channels × 32 output channels × 3×3 kernel + 32 bias = 896
    conv1 = 3 * 32 * 3 * 3 + 32
    # Conv2: 32 input channels × 64 output channels × 3×3 kernel + 64 bias = 18496
    conv2 = 32 * 64 * 3 * 3 + 64
    # FC1: 64×8×8 = 4096 inputs × 128 outputs + 128 bias = 524416
    fc1 = 4096 * 128 + 128
    # FC2: 128 inputs × 10 outputs + 10 bias = 1290
    fc2 = 128 * 10 + 10
    
    total = conv1 + conv2 + fc1 + fc2
    return total


def generate_simplecnn_table():
    """Generate LaTeX table for SimpleCNN architecture."""
    total_params = count_parameters_simplecnn()
    lines = []
    
    lines.append("% SimpleCNN Architecture")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{SimpleCNN architecture: layer configuration, output shape, and parameters.}")
    lines.append("  \\label{tab:simplecnn_arch}")
    lines.append("  \\begin{tabular}{l c c c}")
    lines.append("    \\toprule")
    lines.append("    Layer & Kernel/Units & Output Shape & Parameters \\\\")
    lines.append("    \\midrule")
    
    # Layer details
    layers = [
        ("Conv2d-1", "3 → 32, 3×3", "32×32×32", "32×3×3×3 + 32 = 896"),
        ("ReLU-1", "—", "32×32×32", "0"),
        ("MaxPool2d-1", "2×2, stride=2", "32×16×16", "0"),
        ("Conv2d-2", "32 → 64, 3×3", "64×16×16", "64×32×3×3 + 64 = 18,496"),
        ("ReLU-2", "—", "64×16×16", "0"),
        ("MaxPool2d-2", "2×2, stride=2", "64×8×8", "0"),
        ("Flatten", "—", "4,096", "0"),
        ("Linear-1", "4,096 → 128", "128", "4,096×128 + 128 = 524,416"),
        ("ReLU-3", "—", "128", "0"),
        ("Dropout (0.2)", "—", "128", "0"),
        ("Linear-2", "128 → 10", "10", "128×10 + 10 = 1,290"),
    ]
    
    for layer, config, output, params in layers:
        lines.append(f"    {layer} & {config} & {output} & {params} \\\\")
    
    lines.append("    \\midrule")
    lines.append(f"    \\textbf{{Total}} & — & — & \\textbf{{{total_params:,}}} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)


def generate_architecture_content():
    """Generate body-only content with architecture tables."""
    total_params = count_parameters_simplecnn()
    lines = []
    
    lines.append("% SimpleCNN Architecture")
    lines.append("\\begin{table}[t]")
    lines.append("  \\centering")
    lines.append("  \\caption{SimpleCNN architecture: layer configuration, output shape, and parameters.}")
    lines.append("  \\label{tab:simplecnn_arch}")
    lines.append("  \\begin{tabular}{l c c c}")
    lines.append("    \\toprule")
    lines.append("    Layer & Kernel/Units & Output Shape & Parameters \\\\")
    lines.append("    \\midrule")
    
    layers = [
        ("Conv2d-1", "3 → 32, 3×3", "32×32×32", "896"),
        ("ReLU-1", "—", "32×32×32", "0"),
        ("MaxPool2d-1", "2×2, stride=2", "32×16×16", "0"),
        ("Conv2d-2", "32 → 64, 3×3", "64×16×16", "18,496"),
        ("ReLU-2", "—", "64×16×16", "0"),
        ("MaxPool2d-2", "2×2, stride=2", "64×8×8", "0"),
        ("Flatten", "—", "4,096", "0"),
        ("Linear-1", "4,096 → 128", "128", "524,416"),
        ("ReLU-3", "—", "128", "0"),
        ("Dropout (0.2)", "—", "128", "0"),
        ("Linear-2", "128 → 10", "10", "1,290"),
    ]
    
    for layer, config, output, params in layers:
        lines.append(f"    {layer} & {config} & {output} & {params} \\\\")
    
    lines.append("    \\midrule")
    lines.append(f"    \\textbf{{Total}} & — & — & \\textbf{{{total_params:,}}} \\\\")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    
    return "\n".join(lines)


if __name__ == '__main__':
    import os
    out_dir = os.path.join('paper')
    os.makedirs(out_dir, exist_ok=True)
    
    # Write standalone table
    content = generate_simplecnn_table()
    out_path = os.path.join(out_dir, 'simplecnn_arch.tex')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Wrote {out_path}")
    
    # Write body-only content
    body_content = generate_architecture_content()
    body_out_path = os.path.join(out_dir, 'simplecnn_arch_content.tex')
    with open(body_out_path, 'w', encoding='utf-8') as f:
        f.write(body_content)
    print(f"Wrote {body_out_path}")
