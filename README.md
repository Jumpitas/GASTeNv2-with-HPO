# GASTeNv2-with-HPO
*Generative Adversarial Stress-Test Networks + Hyper-Parameter Optimisation*

![License](https://img.shields.io/static/v1?label=license&message=CC-BY-NC-ND-4.0&color=green)

`GASTeNv2-with-HPO` builds on the original **GASTeN** and adds:

* **Gaussian ambiguity loss (v2)**
* Two-stage **SMAC** hyper-parameter optimisation
* A pluggable **classifier zoo** (CNN, MLP, ResNet-18, ViT-Small, ConvNeXt-Tiny …)
* Community contributions  
  * [Catia’s GASTeNv2 fork](https://github.com/catianag/GASTeNv2) — Gaussian-loss ideas  
  * [Luuk’s GASTeN-HPO](https://github.com/luukgr/GASTeN-HPO) — early SMAC sweeps  

The trained GAN **generates realistic boundary samples** that expose classifier blind-spots and often fool humans.

---

## 1 · Quick Start

```bash
# 1 — clean Python 3.10 environment
mamba create -n gasten python=3.10      # or conda / venv
mamba activate gasten
pip install -r requirements.txt         # torch, timm, smac, wandb, …

# 2 — minimal .env  (adjust paths!)
cat > .env <<'EOF'
CUDA_VISIBLE_DEVICES=0
FILESDIR=/abs/path/for/large_files      # ≈30 GB free
ENTITY=my-wandb-team
EOF

# 3 — fetch a dataset (example: MNIST)
python -m src.data_loaders --download mnist --root $FILESDIR/data
````

---

## 2 · Preparation Pipeline

| Step | Purpose                                            | Example command                                                                                                       |
| ---- | -------------------------------------------------- |-----------------------------------------------------------------------------------------------------------------------|
| 1    | **FID reference stats** for *all* class pairs      | `python src/gen_pairwise_inception.py --dataset mnist`                                                                |
| 1′   | Stats for one pair (7 v 1)                         | `python -m src.metrics.fid --data $FILESDIR/data --dataset mnist --pos 7 --neg 1`                                     |
| 2    | Train a **classifier** (CNN / MLP / ResNet / …)    | `python src/gen_classifiers.py --classifier-type cnn --pos 7 --neg 1 --epochs 50 --nf 50 --seed 4441 --dataset mnist` |
| 3    | Generate **fixed noise pool** (for FID & coverage) | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                                                                   |

> **Custom CNN widths**: `--nf 4-8,8-16-32,20` ⇒ three CNNs of depth 2, 3 and 1.

---

## 3 · Two-Stage HPO Training

| Stage      | Goal                                         | Entry-point                                              | Minimal run                                                                                                                |
| ---------- | -------------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Step-1** | Find a base GAN (good FID)                   | `src/optimization/gasten_bayesian_optimization_step1.py` | <br>`bash\npython -m src.optimization.gasten_bayesian_optimization_step1 \\\n  --config experiments/mnist-7v1_step1.yml\n` |
| **Step-2** | Fine-tune with Gaussian loss (search α, σ …) | `src/optimization/gasten_bayesian_optimization_step2.py` | <br>`bash\npython -m src.optimization.gasten_bayesian_optimization_step2 \\\n  --config experiments/mnist-7v1_step2.yml\n` |

Each stage writes a **JSON** (best hyper-params) and a **TXT** summary (FID & Ambiguity CD).

Swap YAMLs to target different datasets, networks or HPO budgets.

---

## 4 · Supported Datasets

| Dataset         | Resolution | Channels | Notes            |
| --------------- | ---------- | -------- | ---------------- |
| MNIST           | 28 × 28    | 1        | baseline         |
| Fashion-MNIST   | 28 × 28    | 1        | harder textures  |
| CIFAR-10        | 32 × 32    | 3        | colour           |
| STL-10          | 96 × 96    | 3        | larger, few-shot |
| ChestXRay       | 128 × 128  | 3        | medical          |
| ImageNet subset | 224 × 224  | 3        | scalability demo |

---

## 5 · Highlights

* **Gaussian ambiguity loss** (α, σ tuned by SMAC)
* Two-stage **HPO** (learning rate, β values, ResBlocks, …)
* **Classifier zoo**: CNN, MLP, ResNet-18 (frozen / finetune), ViT-Small, ConvNeXt-Tiny
* **Deterministic & reproducible** (global + worker seeds)
* **Seamless W\&B logging** (`WANDB_DISABLED=true` to skip)
* Metrics: **FID**, Ambiguity Classifier Distance, Precision / Recall, Hubris, histograms
* Unified checkpoints for easy resume & comparison


---

## 7 · License & Credits

| Item                | Source                                                    |
| ------------------- | --------------------------------------------------------- |
| Code                | **CC BY-NC-ND 4.0** – research & teaching only            |
| Foundation          | [luispcunha/GASTeN](https://github.com/luispcunha/GASTeN) |
| Gaussian loss ideas | [Catia’s GASTeNv2](https://github.com/catianag/GASTeNv2)  |
| SMAC workflow       | [Luuk’s GASTeN-HPO](https://github.com/luukgr/GASTeN-HPO) |
