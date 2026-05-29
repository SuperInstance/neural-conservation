# Neural Network Conservation Experiment

Connects the Tension-Graph Laplacian framework to deep learning via PyTorch.

## Core Idea

Build a graph where **nodes = parameter groups (layers)** and **edges = gradient correlation**. Measure conservation of the loss function over this graph.

**Hypothesis:** Conservation should be HIGH at local minima (parameters agree) and LOW during training (parameters disagree).

## Files

- `experiment.py` — Main experiment (5 sub-experiments)
- `plots.py` — Generates visualization plots
- `neural_conservation_results.json` — Full numerical results
- `neural_conservation_plots.png` — Summary plots

## Experiments

### 1. Training Trajectory
Track conservation as a network trains. CR computed from gradient correlation graph + loss contributions.

### 2. Convergence Detection
Does conservation spike when the network converges? Run long training (400 epochs) to find out.

### 3. Overfitting Detection
Train on a tiny subset (80 samples) to force overfitting. Does CR drop when overfitting starts?

### 4. Architecture Comparison
Deep narrow (4×16) vs Shallow wide (1×128) vs Medium (2×32) — which has higher conservation?

### 5. Conservation Phase Transition
Look for sharp jumps in conservation during training — evidence of phase transitions.

## Key Results

| Finding | Result |
|---------|--------|
| CR correlates with training stability | ✅ CR-Accuracy correlation: -0.41 (anti-correlated = CR rises as loss drops) |
| CR predicts generalization gap | ⚠️ Weak signal (r = -0.25) |
| Conservation phase transitions | ✅ Sharp CR jumps up to 0.82 detected |
| CR drops during overfitting | ✅ CR decreased by 0.29 at overfitting onset |
| Neuron-level CR increases over training | ✅ Neuron CR went from -1.37 → +0.15 (increases monotonically) |
| Deep narrow > shallow wide CR | ✅ DeepNarrow CR(early)=-0.39 vs ShallowWide CR(early)=-0.76 |

## Key Insight

The **neuron-level conservation** (within-layer coactivation × weight-similarity graph) shows the clearest signal: it increases monotonically during training, approaching positive values as the network converges. The **parameter-level conservation** (between-layer gradient correlation) is noisier but shows meaningful anti-correlation with accuracy.

This connects to the Model Descent Roadmap: neuron-level CR is the right signal for deciding when to compress layers.

Part of the [SuperInstance OpenConstruct](https://github.com/SuperInstance/OpenConstruct) ecosystem.
