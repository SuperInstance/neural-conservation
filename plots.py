"""
Generate plots for the neural conservation experiment.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(__file__).parent

with open(base / "neural_conservation_results.json") as f:
    data = json.load(f)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Neural Network Conservation Experiment\nParameter-Graph Conservation over Training", fontsize=14, fontweight='bold')

# Plot 1: Training trajectory - CR and accuracy
ax = axes[0, 0]
h1 = data['experiment_1_trajectory']['history']
epochs = h1['epoch']
ax.plot(epochs, h1['cr_loss'], 'b-', alpha=0.7, label='CR(loss)')
ax.plot(epochs, h1['cr_norms'], 'g-', alpha=0.7, label='CR(norms)')
ax.plot(epochs, h1['avg_neuron_cr'], 'r-', alpha=0.7, label='Neuron CR')
ax.set_xlabel('Epoch')
ax.set_ylabel('Conservation Ratio')
ax.set_title('Exp 1: CR During Training')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: CR vs Accuracy overlay
ax2 = axes[0, 1]
ax2b = ax2.twinx()
ax2.plot(epochs, h1['cr_loss'], 'b-', alpha=0.6, label='CR(loss)')
ax2b.plot(epochs, h1['test_acc'], 'r-', alpha=0.6, label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('CR(loss)', color='b')
ax2b.set_ylabel('Test Accuracy', color='r')
ax2.set_title('Exp 1: CR vs Accuracy')
ax2.grid(True, alpha=0.3)

# Plot 3: Convergence experiment
ax = axes[0, 2]
h2 = data['experiment_2_convergence']['history']
conv_ep = data['experiment_2_convergence']['convergence_epoch']
ax.plot(h2['epoch'], h2['cr_loss'], 'b-', alpha=0.7, label='CR(loss)')
ax.plot(h2['epoch'], h2['avg_neuron_cr'], 'r-', alpha=0.7, label='Neuron CR')
ax.axvline(conv_ep, color='green', linestyle='--', alpha=0.7, label=f'Convergence (ep {conv_ep})')
ax.set_xlabel('Epoch')
ax.set_ylabel('Conservation Ratio')
ax.set_title('Exp 2: CR Over Long Training')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 4: Overfitting detection
ax = axes[1, 0]
h3 = data['experiment_3_overfitting']['history']
ax.plot(h3['epoch'], h3['train_acc'], 'b-', alpha=0.7, label='Train Acc')
ax.plot(h3['epoch'], h3['full_acc'], 'r-', alpha=0.7, label='Full Acc')
ax2 = ax.twinx()
ax2.plot(h3['epoch'], h3['cr_train'], 'g--', alpha=0.6, label='CR(train)')
ax2.plot(h3['epoch'], h3['cr_full'], 'm--', alpha=0.6, label='CR(full)')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax2.set_ylabel('CR(loss)')
ax.set_title('Exp 3: Overfitting Detection')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='center right')
ax.grid(True, alpha=0.3)

# Plot 5: Architecture comparison (bar chart)
ax = axes[1, 1]
arch_data = data['experiment_4_architecture']
names = list(arch_data.keys())
early_crs = [arch_data[n]['mean_cr_loss_early'] for n in names]
late_crs = [arch_data[n]['mean_cr_loss_late'] for n in names]
test_accs = [arch_data[n]['final_test_acc'] for n in names]
x = np.arange(len(names))
w = 0.3
ax.bar(x - w, early_crs, w, label='CR(early)', color='steelblue', alpha=0.7)
ax.bar(x, late_crs, w, label='CR(late)', color='coral', alpha=0.7)
ax.bar(x + w, test_accs, w, label='Test Acc', color='seagreen', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([n.split('-')[0] for n in names], rotation=15, fontsize=9)
ax.set_ylabel('Value')
ax.set_title('Exp 4: Architecture Comparison')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Plot 6: Parameter correlation over training
ax = axes[1, 2]
ax.plot(epochs, h1['param_corr_mean'], 'b-', alpha=0.7, label='Mean |corr|')
ax.fill_between(epochs,
                np.array(h1['param_corr_mean']) - np.array(h1['param_corr_std']),
                np.array(h1['param_corr_mean']) + np.array(h1['param_corr_std']),
                alpha=0.2, color='blue')
ax.set_xlabel('Epoch')
ax.set_ylabel('Gradient Correlation')
ax.set_title('Exp 1: Parameter Correlation')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(base / "neural_conservation_plots.png", dpi=150, bbox_inches='tight')
print(f"Saved: {base / 'neural_conservation_plots.png'}")
