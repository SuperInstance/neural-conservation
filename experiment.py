"""
Neural Network Conservation Experiment
=======================================
Connects the Tension-Graph Laplacian framework to deep learning.

Core hypothesis: A trained neural network's loss landscape, viewed through
a parameter correlation graph, shows conservation that:
  - Is LOW during training (parameters disagree / explore)
  - Is HIGH at local minima (parameters agree / converge)
  - Drops when overfitting starts
  - Varies with architecture

We build a graph where nodes = parameter groups (layers), edges = gradient
correlation, then measure conservation of the loss function over this graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from pathlib import Path
from collections import defaultdict

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================
# Synthetic Data
# ============================================================

def make_spirals(n=2000, noise=0.3):
    n_per = n // 2
    theta = np.linspace(0, 4*np.pi, n_per) + np.random.randn(n_per)*noise
    r = theta/(4*np.pi) + np.random.randn(n_per)*0.1
    X0 = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
    theta2 = np.linspace(0, 4*np.pi, n_per) + np.random.randn(n_per)*noise
    r2 = theta2/(4*np.pi) + np.random.randn(n_per)*0.1
    X1 = np.column_stack([r2*np.cos(theta2+np.pi), r2*np.sin(theta2+np.pi)])
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.array([0]*n_per + [1]*n_per, dtype=np.int64)
    idx = np.random.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])

def make_xor(n=2000):
    X = np.random.randn(n, 2).astype(np.float32) * 1.5
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(np.int64)
    idx = np.random.permutation(len(y))
    return torch.tensor(X[idx]), torch.tensor(y[idx])

# ============================================================
# Parameter Graph & Conservation
# ============================================================

class ParameterGraph:
    """Build a graph over model parameters based on gradient correlations."""

    def __init__(self, model):
        self.named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.names = [n for n, _ in self.named_params]
        self.n = len(self.named_params)

    def compute_gradients(self, model, loss_fn, X, y):
        """Compute per-parameter gradients as flattened vectors."""
        model.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        grads = []
        for name, param in self.named_params:
            if param.grad is not None:
                grads.append(param.grad.clone().detach().flatten())
            else:
                grads.append(torch.zeros(param.numel(), device=param.device))
        return grads, loss.item()

    def build_correlation_matrix(self, grads):
        """Build gradient correlation matrix between parameter groups.
        
        Since parameter groups have different sizes, we use pairwise statistics:
        1. Sign agreement (fraction of elements that agree in sign)
        2. Cosine similarity on random projections to common dimension
        """
        n = len(grads)
        corr = np.zeros((n, n))
        
        # Project all gradients to a common dimension via random hashing
        proj_dim = 128
        rng = np.random.RandomState(42)
        projections = []
        for g in grads:
            g_np = g.cpu().numpy()
            # Random projection to fixed dimension
            if len(g_np) >= proj_dim:
                # Sample random indices
                idx = rng.choice(len(g_np), proj_dim, replace=False)
                projections.append(g_np[idx])
            else:
                # Pad with zeros
                padded = np.zeros(proj_dim)
                padded[:len(g_np)] = g_np
                projections.append(padded)
        
        for i in range(n):
            for j in range(n):
                gi = projections[i]
                gj = projections[j]
                ni = np.linalg.norm(gi) + 1e-10
                nj = np.linalg.norm(gj) + 1e-10
                corr[i, j] = np.dot(gi, gj) / (ni * nj)
        return corr

    def build_transition_matrix(self, corr):
        """Convert correlation matrix to a valid transition matrix (row-stochastic).
        Use absolute correlations as edge weights — we care about magnitude of agreement."""
        W = np.abs(corr)
        np.fill_diagonal(W, 0)
        # Row-normalize
        row_sums = W.sum(axis=1, keepdims=True) + 1e-10
        T = W / row_sums
        return T

    def compute_laplacian(self, T):
        """Graph Laplacian from transition matrix."""
        W = T.copy()
        np.fill_diagonal(W, 0)
        D = np.diag(W.sum(axis=1))
        L = D - W
        return L

    def conservation_ratio(self, L, attribute):
        """
        Conservation ratio: how smooth is the attribute on this graph?

        CR = 1 - Var(∇²f) / Var(f)
        where ∇²f = L @ f (Laplacian applied to attribute)

        High CR → attribute is conserved (smooth on graph)
        Low CR → attribute is not conserved (rough on graph)
        """
        f = np.array(attribute, dtype=float)
        if np.var(f) < 1e-12:
            return 1.0  # constant function = perfectly smooth
        lap_f = L @ f
        var_lap = np.var(lap_f)
        var_f = np.var(f)
        cr = 1.0 - var_lap / var_f
        return float(np.clip(cr, -5, 1.0))

    def spectral_conservation(self, L, attribute):
        """
        Spectral conservation: project attribute onto eigenvectors of L,
        measure fraction of energy in low-frequency modes.
        """
        f = np.array(attribute, dtype=float)
        if L.shape[0] < 2:
            return 1.0
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        projections = eigenvectors.T @ f
        energy = projections ** 2
        total_energy = energy.sum() + 1e-12
        # Fraction in modes with low eigenvalues (smooth modes)
        half = len(eigenvalues) // 2
        low_energy = energy[:half].sum()
        return float(low_energy / total_energy)


class ActivationGraph:
    """Build a neuron-level coactivation graph within a layer."""

    def __init__(self):
        self.coactivation_buffer = []

    def observe(self, activations):
        """Record batch activations. activations: (batch, n_neurons)"""
        self.coactivation_buffer.append(activations.detach().cpu().numpy())

    def build_graph(self, weight_matrix, activations):
        """Build coactivation × weight-similarity graph.
        
        Args:
            weight_matrix: the weight matrix for this layer (in_features, out_features)
            activations: activation tensor (batch, n_neurons)
        """
        all_act = activations.detach().cpu().numpy()
        n_neurons = all_act.shape[1]

        if n_neurons < 3:
            return None

        # Coactivation frequency
        binary = (all_act > 0.1).astype(float)
        coact = binary.T @ binary / all_act.shape[0]

        # Weight cosine similarity between neuron columns
        W = weight_matrix.detach().cpu().numpy()  # (in_features, out_features)
        w_vecs = W.T  # (n_neurons, in_features)

        norms = np.linalg.norm(w_vecs, axis=1, keepdims=True) + 1e-10
        w_normed = w_vecs / norms
        cos_sim = w_normed @ w_normed.T
        np.fill_diagonal(cos_sim, 0)

        # Combined graph
        G = coact * np.maximum(cos_sim, 0)
        return G

    def clear(self):
        self.coactivation_buffer = []


# ============================================================
# Network Architectures
# ============================================================

class DeepNarrow(nn.Module):
    """Deep narrow network: 4 hidden layers of 16 neurons."""
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, out_dim)
        )
        # Store intermediate activations
        self._activations = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 and isinstance(layer, nn.ReLU):
                self._activations[f'layer_{i//2}'] = x.detach()
        return x

    def get_layer_activations(self):
        return self._activations


class ShallowWide(nn.Module):
    """Shallow wide network: 1 hidden layer of 128 neurons."""
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )
        self._activations = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self._activations['layer_0'] = x.detach()
        return x

    def get_layer_activations(self):
        return self._activations


class MediumNet(nn.Module):
    """Medium: 2 hidden layers of 32 neurons."""
    def __init__(self, in_dim=2, out_dim=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, out_dim)
        )
        self._activations = {}

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                self._activations[f'layer_{i//2}'] = x.detach()
        return x

    def get_layer_activations(self):
        return self._activations


# ============================================================
# Experiment Runners
# ============================================================

def track_conservation_during_training(model, X_train, y_train, X_test, y_test,
                                        epochs=200, lr=0.01, batch_size=64,
                                        experiment_name="default"):
    """
    Train a model and track parameter-graph conservation at every epoch.
    """
    model = model.to(DEVICE)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    param_graph = ParameterGraph(model)
    activation_graph = ActivationGraph()
    
    n = len(y_train)
    history = defaultdict(list)
    
    print(f"\n  Training {experiment_name} ({sum(p.numel() for p in model.parameters())} params)...")
    
    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(n)
        epoch_loss = 0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            bi = idx[start:end]
            Xb, yb = X_train[bi], y_train[bi]
            
            optimizer.zero_grad()
            output = model(Xb)
            loss = loss_fn(output, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            # Observe activations for neuron-level graph
            if hasattr(model, 'get_layer_activations'):
                acts = model.get_layer_activations()
                for name, act in acts.items():
                    activation_graph.observe(act)
        
        # Evaluate conservation every epoch
        model.eval()
        with torch.no_grad():
            train_out = model(X_train)
            train_acc = (train_out.argmax(1) == y_train).float().mean().item()
            train_loss = loss_fn(train_out, y_train).item()
            
            test_out = model(X_test)
            test_acc = (test_out.argmax(1) == y_test).float().mean().item()
            test_loss = loss_fn(test_out, y_test).item()
        
        # Build parameter graph from current gradients
        model.train()
        grads, current_loss = param_graph.compute_gradients(model, loss_fn, X_train[:200], y_train[:200])
        corr = param_graph.build_correlation_matrix(grads)
        T = param_graph.build_transition_matrix(corr)
        L = param_graph.compute_laplacian(T)
        
        # Attribute: loss contribution of each parameter group
        with torch.no_grad():
            param_loss = []
            output = model(X_train[:200])
            loss_val = loss_fn(output, y_train[:200])
            # Use gradient magnitude as proxy for loss contribution
            for g in grads:
                param_loss.append(float(torch.norm(g).item()))
        
        cr_loss = param_graph.conservation_ratio(L, param_loss)
        spectral_cr = param_graph.spectral_conservation(L, param_loss)
        
        # Attribute: parameter norms
        param_norms = [float(p.norm().item()) for _, p in param_graph.named_params]
        cr_norms = param_graph.conservation_ratio(L, param_norms)
        
        # Neuron-level conservation (from activation graph)
        neuron_cr = {}
        if hasattr(model, 'get_layer_activations'):
            with torch.no_grad():
                _ = model(X_train[:200])
            acts = model.get_layer_activations()
            
            # Match activations to weight matrices by layer index
            weight_layers = [(n, p) for n, p in model.named_parameters() if 'weight' in n and p.dim() == 2]
            act_keys = sorted(acts.keys())
            
            for act_idx, act_name in enumerate(act_keys):
                if act_idx < len(weight_layers):
                    wname, wp = weight_layers[act_idx]
                    act = acts[act_name]
                    # Ensure dimensions match
                    if wp.shape[1] == act.shape[1]:
                        G = activation_graph.build_graph(wp, act)
                    elif wp.shape[0] == act.shape[1]:
                        G = activation_graph.build_graph(wp.T, act)
                    else:
                        continue
                    
                    if G is not None and G.shape[0] >= 3:
                        nG = G.shape[0]
                        DL = np.diag(G.sum(axis=1)) - G
                        mean_act = act.mean(dim=0).cpu().numpy()[:nG]
                        if np.var(mean_act) > 1e-12:
                            lap_act = DL @ mean_act
                            neuron_cr[act_name] = 1.0 - np.var(lap_act) / (np.var(mean_act) + 1e-12)
                        else:
                            neuron_cr[act_name] = 1.0
        
        activation_graph.clear()
        
        # Log
        avg_neuron_cr = np.mean(list(neuron_cr.values())) if neuron_cr else 0.0
        
        history['epoch'].append(epoch)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['cr_loss'].append(cr_loss)
        history['cr_norms'].append(cr_norms)
        history['spectral_cr'].append(spectral_cr)
        history['avg_neuron_cr'].append(avg_neuron_cr)
        history['param_corr_mean'].append(float(np.mean(np.abs(corr[np.triu_indices_from(corr, k=1)]))))
        history['param_corr_std'].append(float(np.std(corr[np.triu_indices_from(corr, k=1)])))
        
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}: loss={train_loss:.4f} "
                  f"train={train_acc:.3f} test={test_acc:.3f} "
                  f"CR(loss)={cr_loss:.4f} CR(norms)={cr_norms:.4f} "
                  f"neuron_CR={avg_neuron_cr:.4f}")
    
    return dict(history)


def overfitting_experiment(model_fn, X_train, y_train, n_train_small=100, epochs=500, lr=0.005):
    """
    Experiment: does conservation detect overfitting?
    Train on a small subset (will overfit), track conservation on train vs test.
    """
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    
    # Use small training set to force overfitting
    idx = torch.randperm(len(y_train))[:n_train_small]
    X_small, y_small = X_train[idx], y_train[idx]
    
    # Full test set
    X_test = X_train  # use full dataset as "test" proxy
    y_test = y_train
    
    model = model_fn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    param_graph = ParameterGraph(model)
    history = defaultdict(list)
    
    print(f"\n  Overfitting experiment (n_train={n_train_small}, epochs={epochs})...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_small)
        loss = loss_fn(output, y_small)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            train_acc = (model(X_small).argmax(1) == y_small).float().mean().item()
            full_acc = (model(X_test).argmax(1) == y_test).float().mean().item()
            train_loss = loss_fn(model(X_small), y_small).item()
            full_loss = loss_fn(model(X_test), y_test).item()
        
        # Conservation on training gradients
        model.train()
        grads, _ = param_graph.compute_gradients(model, loss_fn, X_small, y_small)
        corr = param_graph.build_correlation_matrix(grads)
        T = param_graph.build_transition_matrix(corr)
        L = param_graph.compute_laplacian(T)
        
        param_loss = [float(torch.norm(g).item()) for g in grads]
        cr_train = param_graph.conservation_ratio(L, param_loss)
        
        # Conservation on full-set gradients
        grads_full, _ = param_graph.compute_gradients(model, loss_fn, X_test[:500], y_test[:500])
        corr_full = param_graph.build_correlation_matrix(grads_full)
        T_full = param_graph.build_transition_matrix(corr_full)
        L_full = param_graph.compute_laplacian(T_full)
        
        param_loss_full = [float(torch.norm(g).item()) for g in grads_full]
        cr_full = param_graph.conservation_ratio(L_full, param_loss_full)
        
        generalization_gap = train_acc - full_acc
        
        history['epoch'].append(epoch)
        history['train_acc'].append(train_acc)
        history['full_acc'].append(full_acc)
        history['train_loss'].append(train_loss)
        history['full_loss'].append(full_loss)
        history['cr_train'].append(cr_train)
        history['cr_full'].append(cr_full)
        history['gen_gap'].append(generalization_gap)
        
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch:3d}: train={train_acc:.3f} full={full_acc:.3f} "
                  f"gap={generalization_gap:.3f} "
                  f"CR(train)={cr_train:.4f} CR(full)={cr_full:.4f}")
    
    return dict(history)


def architecture_comparison(archs, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    """Compare conservation across architectures."""
    results = {}
    for name, model_fn in archs:
        model = model_fn()
        n_params = sum(p.numel() for p in model.parameters())
        hist = track_conservation_during_training(
            model, X_train, y_train, X_test, y_test,
            epochs=epochs, lr=lr, experiment_name=name
        )
        results[name] = {
            'history': hist,
            'n_params': n_params,
            'final_test_acc': hist['test_acc'][-1],
            'final_cr_loss': hist['cr_loss'][-1],
            'final_cr_norms': hist['cr_norms'][-1],
            'final_neuron_cr': hist['avg_neuron_cr'][-1],
            'max_cr_loss': max(hist['cr_loss']),
            'mean_cr_loss_late': np.mean(hist['cr_loss'][-20:]),
            'mean_cr_loss_early': np.mean(hist['cr_loss'][:20]),
        }
        print(f"    → {name}: {n_params} params, test_acc={results[name]['final_test_acc']:.4f}, "
              f"CR(early)={results[name]['mean_cr_loss_early']:.4f}, "
              f"CR(late)={results[name]['mean_cr_loss_late']:.4f}")
    return results


# ============================================================
# Analysis & Visualization
# ============================================================

def print_trajectory_analysis(name, hist):
    """Analyze conservation trajectory during training."""
    cr = hist['cr_loss']
    test_acc = hist['test_acc']
    
    # Find when conservation peaks
    peak_epoch = np.argmax(cr)
    peak_cr = cr[peak_epoch]
    
    # Find when accuracy plateaus (within 1% of final)
    final_acc = test_acc[-1]
    plateau_epoch = next((i for i, a in enumerate(test_acc) if a >= final_acc - 0.01), len(test_acc)-1)
    
    # Conservation trend
    early_cr = np.mean(cr[:20])
    mid_cr = np.mean(cr[len(cr)//3:2*len(cr)//3])
    late_cr = np.mean(cr[-20:])
    
    # Correlation between CR and accuracy
    cr_arr = np.array(cr)
    acc_arr = np.array(test_acc)
    if np.std(cr_arr) > 1e-6 and np.std(acc_arr) > 1e-6:
        corr_cr_acc = np.corrcoef(cr_arr, acc_arr)[0, 1]
    else:
        corr_cr_acc = 0.0
    
    print(f"\n  📊 {name} Conservation Trajectory:")
    print(f"     CR early={early_cr:.4f}  mid={mid_cr:.4f}  late={late_cr:.4f}")
    print(f"     CR peak={peak_cr:.4f} at epoch {peak_epoch}")
    print(f"     Accuracy plateau at epoch {plateau_epoch} (acc={final_acc:.4f})")
    print(f"     CR-Accuracy correlation: {corr_cr_acc:.4f}")
    
    return {
        'early_cr': early_cr, 'mid_cr': mid_cr, 'late_cr': late_cr,
        'peak_cr': peak_cr, 'peak_epoch': int(peak_epoch),
        'plateau_epoch': int(plateau_epoch),
        'cr_acc_correlation': float(corr_cr_acc),
    }


def print_overfitting_analysis(hist):
    """Analyze whether conservation detects overfitting."""
    cr_train = hist['cr_train']
    cr_full = hist['cr_full']
    gen_gap = hist['gen_gap']
    train_acc = hist['train_acc']
    full_acc = hist['full_acc']
    
    # When does overfitting start? (gen gap exceeds 5%)
    overfit_start = next((i for i, g in enumerate(gen_gap) if g > 0.05), len(gen_gap)-1)
    
    # CR behavior around overfitting
    before = max(0, overfit_start - 20)
    after = min(len(cr_train) - 1, overfit_start + 20)
    
    cr_before = np.mean(cr_train[before:overfit_start]) if overfit_start > before else cr_train[0]
    cr_after = np.mean(cr_train[overfit_start:after+1])
    
    # Correlation of CR drop with generalization gap
    cr_arr = np.array(cr_train)
    gap_arr = np.array(gen_gap)
    if np.std(cr_arr) > 1e-6:
        corr = np.corrcoef(cr_arr, gap_arr)[0, 1]
    else:
        corr = 0.0
    
    print(f"\n  📊 Overfitting Detection Analysis:")
    print(f"     Overfitting starts ~epoch {overfit_start} (gap > 5%)")
    print(f"     CR(train) before overfit: {cr_before:.4f}")
    print(f"     CR(train) after overfit:  {cr_after:.4f}")
    print(f"     CR change: {cr_after - cr_before:+.4f}")
    print(f"     CR-GenGap correlation: {corr:.4f}")
    print(f"     Final train_acc={train_acc[-1]:.4f}  full_acc={full_acc[-1]:.4f}")
    
    return {
        'overfit_start': int(overfit_start),
        'cr_before': cr_before,
        'cr_after': cr_after,
        'cr_change': cr_after - cr_before,
        'cr_gap_corr': float(corr),
    }


# ============================================================
# MAIN EXPERIMENTS
# ============================================================

print("=" * 70)
print("NEURAL NETWORK CONSERVATION EXPERIMENT")
print("Connecting Tension-Graph Laplacians to Deep Learning")
print("=" * 70)

# Generate data
print("\n📊 Generating data...")
X, y = make_spirals(n=2000)
split = 1600
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"  Spirals: {len(y_train)} train, {len(y_test)} test")

# ============================================================
# EXPERIMENT 1: Training Trajectory
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: CONSERVATION OVER TRAINING TRAJECTORY")
print("Hypothesis: CR LOW during training, HIGH at convergence")
print("=" * 70)

model1 = MediumNet()
hist1 = track_conservation_during_training(
    model1, X_train, y_train, X_test, y_test,
    epochs=200, lr=0.01, batch_size=64,
    experiment_name="MediumNet-spirals"
)

traj_analysis = print_trajectory_analysis("MediumNet", hist1)

# ============================================================
# EXPERIMENT 2: Convergence Detection
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: CONSERVATION AS CONVERGENCE SIGNAL")
print("Hypothesis: Conservation spikes when network converges")
print("=" * 70)

# Train longer to see convergence clearly
model2 = MediumNet()
hist2 = track_conservation_during_training(
    model2, X_train, y_train, X_test, y_test,
    epochs=400, lr=0.005, batch_size=64,
    experiment_name="MediumNet-convergence"
)

traj2 = print_trajectory_analysis("MediumNet-long", hist2)

# Check: does CR derivative predict convergence?
cr2 = np.array(hist2['cr_loss'])
acc2 = np.array(hist2['test_acc'])
dcr = np.diff(cr2)

# Find convergence point (accuracy within 0.5% of max)
max_acc = max(acc2)
conv_point = next((i for i, a in enumerate(acc2) if a >= max_acc - 0.005), len(acc2)-1)

print(f"\n  Convergence at epoch {conv_point}, acc={acc2[conv_point]:.4f}")
print(f"  CR at convergence: {cr2[conv_point]:.4f}")
print(f"  CR trend: early={np.mean(cr2[:30]):.4f} → convergence={np.mean(cr2[conv_point-10:conv_point+10]):.4f} → final={np.mean(cr2[-20:]):.4f}")

# ============================================================
# EXPERIMENT 3: Overfitting Detection
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: OVERFITTING DETECTION")
print("Hypothesis: Conservation drops when overfitting starts")
print("=" * 70)

overfit_hist = overfitting_experiment(
    lambda: MediumNet(),
    X, y,
    n_train_small=80,
    epochs=500,
    lr=0.005
)

overfit_analysis = print_overfitting_analysis(overfit_hist)

# ============================================================
# EXPERIMENT 4: Architecture Comparison
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: ARCHITECTURE COMPARISON")
print("Hypothesis: Different architectures show different conservation profiles")
print("=" * 70)

archs = [
    ("DeepNarrow-4x16", lambda: DeepNarrow()),
    ("ShallowWide-1x128", lambda: ShallowWide()),
    ("Medium-2x32", lambda: MediumNet()),
]

arch_results = architecture_comparison(archs, X_train, y_train, X_test, y_test,
                                        epochs=200, lr=0.01)

print(f"\n{'Architecture':<20} {'Params':>7} {'TestAcc':>8} {'CR(early)':>10} {'CR(late)':>10} {'ΔCR':>8}")
print("-" * 65)
for name, res in arch_results.items():
    delta = res['mean_cr_loss_late'] - res['mean_cr_loss_early']
    print(f"{name:<20} {res['n_params']:>7} {res['final_test_acc']:>8.4f} "
          f"{res['mean_cr_loss_early']:>10.4f} {res['mean_cr_loss_late']:>10.4f} {delta:>+8.4f}")

# ============================================================
# EXPERIMENT 5: Conservation Phase Transition
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT 5: CONSERVATION PHASE TRANSITION")
print("Looking for sharp transitions in CR during training")
print("=" * 70)

model5 = DeepNarrow()
hist5 = track_conservation_during_training(
    model5, X_train, y_train, X_test, y_test,
    epochs=300, lr=0.01, batch_size=64,
    experiment_name="DeepNarrow-phase"
)

cr5 = np.array(hist5['cr_loss'])
dcr5 = np.abs(np.diff(cr5))
# Find top jumps
top_jumps = np.argsort(dcr5)[-5:][::-1]
print(f"\n  Top 5 CR jumps:")
for rank, idx in enumerate(top_jumps):
    print(f"    #{rank+1}: Epoch {idx}→{idx+1}, CR {cr5[idx]:.4f}→{cr5[idx+1]:.4f} "
          f"(Δ={dcr5[idx]:.4f}), acc={hist5['test_acc'][idx]:.4f}")

# Check if jumps coincide with accuracy changes
acc5 = np.array(hist5['test_acc'])
dacc5 = np.abs(np.diff(acc5))
cr_acc_jump_corr = np.corrcoef(dcr5, dacc5)[0, 1] if np.std(dcr5) > 1e-6 else 0
print(f"  Correlation between CR jumps and accuracy jumps: {cr_acc_jump_corr:.4f}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("🔬 FINAL SUMMARY & KEY FINDINGS")
print("=" * 70)

print("""
EXPERIMENT 1 — Training Trajectory:
""")
print(f"  CR early={traj_analysis['early_cr']:.4f} → mid={traj_analysis['mid_cr']:.4f} → late={traj_analysis['late_cr']:.4f}")
print(f"  CR-Accuracy correlation: {traj_analysis['cr_acc_correlation']:.4f}")
if traj_analysis['late_cr'] > traj_analysis['early_cr']:
    print(f"  ✅ CR INCREASES during training (parameters agree more as they converge)")
else:
    print(f"  ⚠️  CR does NOT increase monotonically")

print("""
EXPERIMENT 2 — Convergence Detection:
""")
print(f"  CR at convergence point: {cr2[conv_point]:.4f}")
if np.mean(cr2[conv_point:]) > np.mean(cr2[:conv_point]):
    print(f"  ✅ CR is HIGHER after convergence")
else:
    print(f"  ⚠️  CR not clearly higher post-convergence")

print("""
EXPERIMENT 3 — Overfitting Detection:
""")
print(f"  Overfitting starts at epoch {overfit_analysis['overfit_start']}")
print(f"  CR change at overfitting: {overfit_analysis['cr_change']:+.4f}")
if overfit_analysis['cr_change'] < 0:
    print(f"  ✅ CR DROPS when overfitting starts")
else:
    print(f"  ⚠️  CR does not clearly drop at overfitting onset")
print(f"  CR-Generalization gap correlation: {overfit_analysis['cr_gap_corr']:.4f}")

print("""
EXPERIMENT 4 — Architecture Comparison:
""")
best_arch = max(arch_results.items(), key=lambda x: x[1]['final_test_acc'])
best_cr = max(arch_results.items(), key=lambda x: x[1]['mean_cr_loss_late'])
print(f"  Best accuracy: {best_arch[0]} ({best_arch[1]['final_test_acc']:.4f})")
print(f"  Highest late-stage CR: {best_cr[0]} ({best_cr[1]['mean_cr_loss_late']:.4f})")

print("""
EXPERIMENT 5 — Phase Transitions:
""")
print(f"  Largest CR jump at epoch {top_jumps[0]} (Δ={dcr5[top_jumps[0]]:.4f})")
print(f"  CR jump ↔ accuracy jump correlation: {cr_acc_jump_corr:.4f}")

# Key questions
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY QUESTIONS ANSWERED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

q1 = traj_analysis['cr_acc_correlation']
print(f"1. Does conservation correlate with training stability?")
print(f"   CR-Accuracy correlation: {q1:.4f}")
print(f"   {'✅ YES — higher CR tracks with better accuracy' if abs(q1) > 0.3 else '⚠️  WEAK — relationship needs more investigation'}")

print(f"\n2. Can conservation predict generalization gap?")
print(f"   CR-GenGap correlation: {overfit_analysis['cr_gap_corr']:.4f}")
print(f"   {'✅ YES — CR tracks generalization' if abs(overfit_analysis['cr_gap_corr']) > 0.3 else '⚠️  WEAK — marginal signal'}")

print(f"\n3. Is there a 'conservation phase transition' during training?")
biggest_jump = dcr5[top_jumps[0]]
print(f"   Largest CR jump: {biggest_jump:.4f}")
print(f"   {'✅ YES — sharp transitions detected' if biggest_jump > 0.1 else '⚠️  WEAK — transitions are gradual'}")


# ============================================================
# Save all results
# ============================================================
output = {
    'experiment_1_trajectory': {
        'history': hist1,
        'analysis': traj_analysis,
    },
    'experiment_2_convergence': {
        'history': hist2,
        'convergence_epoch': int(conv_point),
        'cr_at_convergence': float(cr2[conv_point]),
    },
    'experiment_3_overfitting': {
        'history': overfit_hist,
        'analysis': overfit_analysis,
    },
    'experiment_4_architecture': {
        name: {k: v for k, v in res.items() if k != 'history'}
        for name, res in arch_results.items()
    },
    'experiment_5_phase_transition': {
        'top_jumps': [(int(i), float(dcr5[i]), float(cr5[i])) for i in top_jumps],
        'cr_acc_jump_corr': float(cr_acc_jump_corr),
    },
}

output_path = Path(__file__).parent / "neural_conservation_results.json"
def convert(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(obj)}")

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=convert)

print(f"\n💾 Results saved to: {output_path}")
print("\nDone!")
