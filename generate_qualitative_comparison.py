"""
Generate qualitative comparison figure for paper:
Show sample CIFAR images with Adam vs SR-Adam predictions at noise level 0.05
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import json
import argparse

# Import model architecture
from model import get_model


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('test_acc', None)


def get_best_run(optimizer_name, dataset='CIFAR10', model_name='simplecnn', noise=0.05, runs_root='runs'):
    """Find the run with highest best accuracy for given optimizer."""
    folder = Path(runs_root) / dataset / model_name / f'noise_{noise}' / optimizer_name
    
    if not folder.exists():
        folder = Path('results') / dataset / model_name / f'noise_{noise}' / optimizer_name
    
    if not folder.exists():
        raise FileNotFoundError(f"No results found for {optimizer_name} in {folder}")
    
    # Find all best checkpoints
    best_files = list(folder.glob('run_*_best.pt'))
    
    if not best_files:
        raise FileNotFoundError(f"No best checkpoints found in {folder}")
    
    # Load all and find highest accuracy
    best_acc = -1
    best_path = None
    
    for f in best_files:
        try:
            ckpt = torch.load(f, map_location='cpu')
            acc = ckpt.get('test_acc', -1)
            if acc > best_acc:
                best_acc = acc
                best_path = f
        except:
            continue
    
    return best_path, best_acc


def get_cifar_classes(dataset_name):
    """Return class names for CIFAR datasets."""
    if dataset_name == 'CIFAR10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']
    else:  # CIFAR100
        return [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
            'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
            'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
            'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
            'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
            'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
            'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
            'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
        ]


def generate_comparison_table(dataset='CIFAR10', noise=0.05, num_samples=10, 
                               save_path='paper_figures/qualitative_comparison.pdf',
                               runs_root='runs', random_seed=None):
    """
    Generate comparison table showing sample images with Adam vs SR-Adam predictions.
    
    Args:
        dataset: 'CIFAR10' or 'CIFAR100'
        noise: noise level (0.0, 0.05, 0.1)
        num_samples: number of images to show
        save_path: where to save the figure
        runs_root: root directory for model checkpoints
        random_seed: seed for sample selection (None = random each time)
    """
    # Set random seed for reproducible sample selection
    if random_seed is not None:
        np.random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load class names
    class_names = get_cifar_classes(dataset)
    num_classes = len(class_names)
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset == 'CIFAR10':
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                download=True, transform=transform)
    else:
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                 download=True, transform=transform)
    
    # Find best checkpoints for both optimizers
    print("Loading best models...")
    adam_path, adam_acc = get_best_run('Adam', dataset, 'simplecnn', noise, runs_root)
    sradam_path, sradam_acc = get_best_run('SR-Adam', dataset, 'simplecnn', noise, runs_root)
    
    print(f"\n{'='*60}")
    print(f"LOADED CHECKPOINT ACCURACIES (from best.pt files):")
    print(f"{'='*60}")
    print(f"Adam best checkpoint:    {adam_acc:.2f}%")
    print(f"SR-Adam best checkpoint: {sradam_acc:.2f}%")
    print(f"{'='*60}")
    print(f"Adam best: {adam_path}")
    print(f"SR-Adam best: {sradam_path}")
    print(f"{'='*60}\n")
    
    # Load models
    adam_model = get_model('simplecnn', num_classes=num_classes).to(device)
    sradam_model = get_model('simplecnn', num_classes=num_classes).to(device)
    
    adam_model, _ = load_checkpoint(adam_path, adam_model, device)
    sradam_model, _ = load_checkpoint(sradam_path, sradam_model, device)
    
    # Select diverse samples (try to get variety of classes)
    indices = []
    classes_seen = set()
    
    for idx in np.random.permutation(len(testset)):
        _, label = testset[idx]
        if label not in classes_seen or len(indices) < num_samples:
            indices.append(idx)
            classes_seen.add(label)
        if len(indices) >= num_samples:
            break
    
    # Create figure: TRANSPOSED layout (3 rows × num_samples columns)
    # Rows: True Label, Adam, SR-Adam
    # Columns: Sample images
    fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 7))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"\nGenerating predictions for {num_samples} samples...")
    
    for col, idx in enumerate(indices):
        image, true_label = testset[idx]
        
        # Predict with both models
        with torch.no_grad():
            img_batch = image.unsqueeze(0).to(device)
            
            adam_out = adam_model(img_batch)
            adam_pred = adam_out.argmax(dim=1).item()
            adam_conf = torch.softmax(adam_out, dim=1).max().item()
            
            sradam_out = sradam_model(img_batch)
            sradam_pred = sradam_out.argmax(dim=1).item()
            sradam_conf = torch.softmax(sradam_out, dim=1).max().item()
        
        # Denormalize image for display
        img_np = image.numpy().transpose(1, 2, 0)
        img_np = img_np * 0.5 + 0.5  # denormalize
        img_np = np.clip(img_np, 0, 1)
        
        # Row 0: Ground Truth
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f'{class_names[true_label]}', fontsize=10, fontweight='bold')
        axes[0, col].axis('off')
        
        # Row 1: Adam prediction
        adam_color = 'green' if adam_pred == true_label else 'red'
        axes[1, col].imshow(img_np)
        axes[1, col].set_title(f'{class_names[adam_pred]}\n({adam_conf:.2f})', 
                               color=adam_color, fontsize=9, fontweight='bold')
        axes[1, col].axis('off')
        
        # Row 2: SR-Adam prediction
        sradam_color = 'green' if sradam_pred == true_label else 'red'
        axes[2, col].imshow(img_np)
        axes[2, col].set_title(f'{class_names[sradam_pred]}\n({sradam_conf:.2f})', 
                               color=sradam_color, fontsize=9, fontweight='bold')
        axes[2, col].axis('off')
    
    # Add row labels on the left
    fig.text(0.02, 0.75, 'Ground\nTruth', ha='center', va='center', fontsize=12, fontweight='bold', rotation=0)
    fig.text(0.02, 0.50, f'Adam\n({adam_acc:.1f}%)', ha='center', va='center', fontsize=12, fontweight='bold', rotation=0)
    fig.text(0.02, 0.25, f'SR-Adam\n({sradam_acc:.1f}%)', ha='center', va='center', fontsize=12, fontweight='bold', rotation=0)
    
    plt.tight_layout(rect=[0.05, 0, 1, 1])
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison figure to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser(description='Generate qualitative comparison figures for paper')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--noise', type=float, default=0.05, choices=[0.0, 0.05, 0.1],
                        help='Noise level (default: 0.05)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for sample selection (default: None=random). Try different seeds like 42, 123, 999 to find best visualization')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to display (default: 10)')
    parser.add_argument('--runs-root', type=str, default='runs',
                        help='Root directory for model checkpoints (default: runs)')
    
    args = parser.parse_args()
    
    dataset = args.dataset
    noise = args.noise
    seed = args.seed
    
    # Print aggregated summary values for verification
    print("\n" + "="*70)
    print(f"AGGREGATED SUMMARY (from results/.../aggregated_summary.json):")
    print(f"Dataset: {dataset}, Noise: {noise}")
    if seed is not None:
        print(f"Random Seed: {seed}")
    print("="*70)
    
    summary_data = {}
    for opt in ['Adam', 'SR-Adam']:
        json_path = f'results/{dataset}/simplecnn/noise_{noise}/{opt}/aggregated_summary.json'
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"\n{opt}:")
            print(f"  Final: {data['final_test_acc_mean']:.2f} ± {data['final_test_acc_std']:.2f}%")
            print(f"  Best:  {data['best_test_acc_mean']:.2f} ± {data['best_test_acc_std']:.2f}%")
            summary_data[opt] = data
    print("="*70 + "\n")
    
    # Generate comparison
    dataset_lower = dataset.lower()
    noise_str = str(noise).replace('.', '')
    save_path = f'paper_figures/qualitative_{dataset_lower}_noise{noise_str}.pdf'
    
    print(f"Generating qualitative comparison figure for {dataset} with noise={noise}...")
    if seed is not None:
        print(f"TIP: Try different seeds (e.g., --seed 42, --seed 123, --seed 999) to find the most informative samples!")
    
    generate_comparison_table(
        dataset=dataset,
        noise=noise,
        num_samples=args.num_samples,
        save_path=save_path,
        runs_root=args.runs_root,
        random_seed=seed
    )
    
    # After generation, load the checkpoint accuracies that were used
    print("\n" + "="*70)
    print("CHECKPOINT ACCURACIES USED IN PDF (from best.pt files):")
    print("="*70)
    
    checkpoint_accs = {}
    for opt in ['Adam', 'SR-Adam']:
        try:
            _, acc = get_best_run(opt, dataset, 'simplecnn', noise, args.runs_root)
            checkpoint_accs[opt] = acc
            print(f"{opt}: {acc:.2f}%")
        except Exception as e:
            print(f"{opt}: Error - {e}")
    
    print("="*70)
    
    # Save comparison to JSON
    comparison = {
        "dataset": dataset,
        "noise": noise,
        "random_seed": seed,
        "aggregated_summary": summary_data,
        "checkpoint_best_selected": checkpoint_accs,
        "note": "checkpoint_best_selected shows the single best run accuracy displayed in PDF"
    }
    
    os.makedirs('paper_figures', exist_ok=True)
    json_name = f'qualitative_accuracy_comparison_{dataset_lower}_noise{noise_str}.json'
    with open(f'paper_figures/{json_name}', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved comparison to paper_figures/{json_name}")
    print(f"\n✓ Figure saved to {save_path}")
    print("\nKEY INSIGHT:")
    print("  - Table shows MEAN ± STD across 5 runs")
    print("  - PDF shows accuracy of the BEST SINGLE RUN (highest checkpoint)")
    print("  - This explains why PDF values differ from table values!")
    if seed is not None:
        print(f"\nUsed seed={seed}. To try different samples, use --seed with another value.")
    print("="*70 + "\n")
