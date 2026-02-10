# Assignment 7: Multi-modal Attacks on Aligned Language Models

## Goal
Explore and evaluate multi-modal attacks on aligned language models, specifically implementing adversarial image perturbations to manipulate vision-language model outputs.

---

## Dataset
For simplicity sake we just use a 512x512 blue background with a centered red square

---

## Design & Implementation

### Attack Design

**Method**: Projected Gradient Descent (PGD) adversarial attack based on [*Are aligned neural networks adversarially aligned?*](https://arxiv.org/abs/2306.15447) (Carlini et al., NeurIPS 2023).

**Threat Model**:
- **White-box targeted attack**: Full access to model architecture, parameters, and gradients
- **Objective**: Force vision-language model to generate specific malicious/misleading text
- **Constraint**: L∞-bounded perturbations (imperceptible to humans)

**Key Design Choices**:

1. **Optimization in Processed Space**
   - Problem: Converting between PIL images and model inputs breaks gradient flow due to non-differentiable preprocessing (resize, normalize)
   - Solution: Optimize perturbations δ directly in the model's preprocessed pixel space
   - Benefit: Preserves gradient flow; δ can be directly applied without reconstruction

2. **Gradient Computation Strategy**
   - Uses `torch.autograd.grad(outputs=loss, inputs=delta)` for explicit gradient extraction
   - Kind of follows approach from [multimodal_injection](https://github.com/ebagdasa/multimodal_injection)
   - Sign-based updates (FGSM-style): `δ ← δ - α × sign(∇δ Loss)`
   - Instead of using p(y | y_t < y), we use a normal loss, assuming that the target sentence is the entire sequence.
   - This is because doing it the other way requires a lot more GPU than available in the free tier of google colab.

3. **Attack Parameters**:
   - **ε (epsilon)**: 16/255 ≈ 0.0627 maximum L∞ perturbation
   - **α (step size)**: 1.6/255 (ε/10)
   - **Iterations**: 50 PGD steps
   - **Random initialization**: δ ~ Uniform[-ε, ε]
   - **Projection**: Clamp δ to [-ε, ε] after each step

**Implementation Steps**:
1. Load SmolVLM-256M-Instruct (4-bit quantized) via HuggingFace transformers
2. Process original image → get pixel_values (preprocessed tensors)
3. Initialize random perturbation δ within ε-ball
4. **Optimization loop** (50 iterations):
   - Compute adversarial pixel_values: `adv = original + δ`
   - Forward pass: compute loss for target text generation
   - Backward: `grad = torch.autograd.grad(loss, δ)`
   - Update: `δ ← δ - α × sign(grad)`
   - Project: `δ ← clip(δ, -ε, ε)`
5. Return adversarial pixel_values
6. Reconstruction isn't perfect because of the transformer architecture, hence we instead use the post processed pixel values for testing.

---

### Defense Design

**Defense 1: Median Filter**
- **Mechanism**: Replace each pixel with median of its neighborhood
- **Parameters**: Kernel sizes 3×3 and 5×5
- **Rationale**: Removes salt-and-pepper noise and small adversarial perturbations while preserving edges

**Defense 2: Gaussian Blur**
- **Mechanism**: Smooth image using Gaussian kernel
- **Parameters**: σ = 1.0 and σ = 2.0
- **Rationale**: Attenuates high-frequency adversarial patterns through low-pass filtering

**Defense Application**:
- For preprocessed pixel_values: denormalize → apply filter → re-normalize → re-process
- For PIL images: apply filter → process through model processor
- Both defenses are non-adaptive (attacker not aware during optimization)

---

## Metrics & Results

### Success Criteria

**Attack Success**: Model generates target text ("It is a disgusting colour") instead of accurate description

**Defense Success**: Model returns to normal/benign output after filtering

### Quantitative Metrics

**Attack Performance**:
- **Perturbation Norms** (from attack_results_eps16.png):
  - L∞: 0.0627 (within budget of 16/255)
  - L2: ~32.1 (measured on full image)
  - L1: ~24,660 (total absolute perturbation)
  - Mean absolute: 0.0314

- **Loss Convergence**:
  - Final loss: Converges over 50 iterations
  - Best loss: Tracked and used for final perturbation
  - Gradient flow: Successfully maintained (fixed "no gradient" issue)

**Defense Performance**

| Defense Method | Kernel/σ | Attack Success Rate | Notes |
|----------------|----------|---------------------|-------|
| None (baseline) | - | ~100% (expected) | Direct adversarial input |
| Median Filter | k=3 | 0%| Mild filtering |
| Median Filter | k=5 | 0%| Stronger filtering, may degrade quality |
| Gaussian Blur | σ=1.0 | 0%| Subtle smoothing |
| Gaussian Blur | σ=2.0 | 0%| Aggressive smoothing |


**Expected Results**:
- Undefended attack: Model generates target text with high probability
- Median filtering: Reduces attack success, stronger for larger kernels
- Gaussian blur: Smooths perturbations, effectiveness scales with σ
- Trade-off: Stronger defenses → lower attack success but degraded image quality

### Brief Analysis

**Attack Observations**:
- PGD successfully optimizes perturbations in processed space
- Gradient flow maintained through entire optimization
- Perturbations stay within imperceptible range (L∞ < 16/255)
- Model generates adversarial text when using exact preprocessed δ

**Defense Observations**:
- Simple input transformations provide baseline robustness
- Median filtering preserves edges better than Gaussian blur
- Both defenses degrade image quality (quality-robustness trade-off)
- Non-adaptive defenses: effectiveness limited against adaptive attacks

---

### Expected Output

1. **Dataset Loading**: Loads 120 images from Synthetic_Dataset, displays distribution
2. **Baseline Test**: Original image → model generates accurate description
3. **PGD Attack** (50 iterations):
   - Progress bar showing loss and perturbation norms
   - Convergence to target text generation
4. **Adversarial Testing**:
   - Direct test with preprocessed adversarial pixel_values (200 tokens)
   - Reconstructed PIL image test for comparison
5. **Defense Evaluation**:
   - Median filter (k=3, k=5) results
   - Gaussian blur (σ=1.0, σ=2.0) results
6. **Saved Outputs**:
   - `attack_results_eps16.png`: Visualization (original, adversarial, perturbation heatmap, loss curve)
   - `adversarial_image_eps16.png`: Adversarial image (approximate reconstruction)

---
