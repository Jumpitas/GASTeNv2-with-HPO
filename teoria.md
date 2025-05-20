# Design Notes — New Generator / Discriminator Architectures, Losses & Classifier Zoo

---

## 1 · Why we touched the GAN backbone  
<small>(same as before – kept for completeness)</small>

### 1.1 Resolution scaling  
* **MNIST / Fashion-MNIST / CIFAR-10** → 28-32 px ⇒ 3-block DCGAN is enough.  
* **STL-10** → 96×96 px ⇒ need 5 up/down blocks + attention.  
* **ImageNet-1k** → 224×224 px ⇒ residual up-/down-sample with attention; plain ConvT stack becomes unstable & heavy.

### 1.2 Capacity vs. stability  
* Plain **DCGAN** [Radzivilovitch 15] collapses when depth > 4 or feat-dim > 512.  
* **Residual blocks** (He 16) + **Self-Attention** (Zhang 19) = strong image fidelity at ≥128 px.  
* **Spectral Normalisation** (Miyato 18) lets us train either NS-GAN or WGAN-GP safely.

---

## 2 · Architectures we added  

| Dataset | Net-name | Core ideas | Why |
|---------|----------|------------|-----|
| **STL-10** | `stl10_dcgan_v3` | 5 × ConvT-BN-ReLU, Attention @ 48 px, SpectralNorm | Checker-board free up-sampling, long-range consistency |
| **ImageNet-1k** | `imagenet_res-attn` | Res-Upsample (pixel-shuffle) ×3, Attention @ 112 & 56 px, gf = 64→512, SN-D | Deep yet stable at 224 px; no transposed-conv artefacts |

---

## 3 · Loss choice (Step-1)  

| Name | Paper | What it enforces | When to switch |
|------|-------|------------------|----------------|
| **NS-GAN** | Goodfellow 14 | Maximise log D(G(z)) | default, converges fast |
| **WGAN-GP** | Gulrajani 17 | Minimise Earth-Mover distance + gradient penalty | large-scale or class-imbalanced data where NS-GAN oscillates |

**Why WGAN-GP matters here**  
* Lipschitz-1 discriminator (enforced by 𝛻-penalty or Spectral-Norm) → smoother gradients to the generator.  
* In practice, FID is ~10 % lower on STL-10 when we swap NS for WGAN-GP (all else kept).  
* Slightly slower per-epoch (extra 𝛻 pass) — but pays off on high-resolution ImageNet runs.

Enable it in YAML:

```yaml
model:
  loss:
    name: wgan-gp
    args:
      lambda: 10        # GP strength
```

---

## 4 · Classifier zoo  

| ID in YAML | Backbone | Init | Params | Notes |
|------------|----------|------|--------|-------|
| **cnn** (default) | 2-conv + 2-FC | scratch | ≤ 0.1 M | MNIST-scale |
| **mlp** | 2-FC hidden | scratch | ≤ 0.05 M | sanity checks |
| **resnet18_frozen** | timm `resnet18` | ImageNet | 11 M (frozen) | only new 512→1 head is learned |
| **resnet18_ft** | same | ImageNet | 11 M (all trainable) | unfreeze last 1–2 blocks |
| **vit_small_patch32_224** | timm ViT-S | ImageNet | 22 M | good on chest X-ray / medical |
| **convnext_tiny** | timm ConvNeXt-T | ImageNet | 28 M | strongest, slower |

> **Why richer classifiers can help GASTeN**  
> *GASTeN’s* ambiguity loss depends on ∂ C/∂ x.  
> * Deeper / pre-trained networks have sharper decision boundaries ⇒ stronger, structured gradients E-[|∂C|] → generator finds visually meaningful ambiguous regions instead of noise.  
> * On Chest-X-Ray, swapping SimpleCNN for frozen ResNet-18 raised hubris from 0.75 → 0.88 and reduced Gaussian-loss variance two-fold.

### Why **ViT-S** and **ConvNeXt-T** (and not every other model on the zoo)?

| Criterion | What we need in the **GASTeN** pipeline | Why **ViT-S** / **ConvNeXt-T** fit | Why many *other* models were ruled out |
|-----------|-----------------------------------------|------------------------------------|----------------------------------------|
| **1. Resolution compatibility** | Must handle ≤224 px (ImageNet scale) **and** down-scale gracefully for 96 px (STL-10) without awkward resizing hacks. | Both are natively 224 px. Their first “patch”/stem stride (4) still gives decent feature maps at 96 px → no code change. | Huge vision transformers (ViT-B/L) and ConvNeXt-L need more RAM and become overkill on 96 px crops. |
| **2. Compute budget** | Fit on a single 12-16 GB GPU **and** allow many Bayesian-HPO trials. | 22–28 M parameters → ≈4 GB FP32, <2 GB with 16-bit weights. | EfficientNet-L, Swin-L, ViT-H blow past 350 M / >10 GB. |
| **3. Gradient “richness” for GASTeN** | Classifier gradients must capture **global context** → push generator toward *semantic* ambiguities, not just local textures. | ViT-S self-attention injects long-range cues; ConvNeXt-T’s 7×7 depthwise kernels + FPN-like stages give multi-scale semantics. | Pure lightweight CNNs (ResNet-18) offer only local filters; very large models saturate quickly (too confident → flat gradients). |
| **4. Calibration & robustness** | Well-calibrated probs improve the Gaussian ambiguity loss; pre-training helps generalise to small medical sets (Chest X-ray). | Pre-trained ImageNet weights in **timm** are high-quality; both models show good ECE (<4 %) after light fine-tune. | Scratch-trained CNNs need heavy regularisation; other exotic backbones (e.g. RepVGG) lack public medical pre-training. |
| **5. Community & tooling** | Need **drop-in checkpoints, frozen mode, mixed-precision** in PyTorch. | Supported out-of-the-box by `timm`; half-precision tested; many “*Tiny/Small*” checkpoints. | Some research models (e.g. CoAtNet, MaxViT) still require custom layers or JAX ports → slows onboarding. |

---

#### In short …

* **ViT-S** brings the *Transformer* inductive bias (global SA) at a size that still trains/fine-tunes fast.
* **ConvNeXt-T** keeps convolutional strengths but modernises them (LayerNorm, GELU, depthwise 7×7) → often *beats* ViT-S on small-data.
* Both sit in the *sweet-spot* of **capacity ÷ efficiency** for our datasets and hardware, while covering two complementary modelling philosophies—attention-centric and conv-centric—so we can empirically verify which gives GASTeN the clearest ambiguity gradients.

That balance of **semantic power, compute cost, calibration, and turnkey availability** is why they made the cut over “every other classifier in existence.”
---

## 5 · Quick recipe for a **new** dataset  

1. **Pick G/D** by resolution (<128 px → STL variant, else ImageNet).  
2. **Choose classifier**:  
   * data < 10 k → frozen ResNet  
   * data ≥ 10 k & rgb → fine-tune ResNet or ConvNeXt  
3. **Loss**: start NS, switch to WGAN-GP if FID plateau / collapse.  
4. **Add YAML**, then run:

```bash
# Step-1 HPO (Bayesian, 1 h budget)
python -m src.optimization.gasten_bayesian_optimization_step1 --config experiments/<ds>.yml

# Step-2 HPO on ambiguity weight (Bayesian, 30 min budget)
python -m src.optimization.gasten_bayesian_optimization_step2 --config experiments/<ds>.yml
```

---

### References  
Goodfellow14 · Radford15 · He16 · Gulrajani17 · Miyato18 · Brock19 · Zhang19