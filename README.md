# GASTeNv2-with-HPO

*Generative Adversarial Stress-Test Networks + Hyper-Parameter Optimisation*

![License](https://img.shields.io/static/v1?label=license\&message=CC-BY-NC-ND-4.0\&color=green)

`GASTeNv2-with-HPO` builds on **GASTeN** and adds:

* Gaussian ambiguity loss (v2)
* Two-stage SMAC hyper-parameter optimisation
* A pluggable classifier zoo (CNN, MLP, ResNet-18, ViT-Small, ConvNeXt-Tiny, …)
* Community contributions:

  * Catia’s [GASTeNv2 fork](https://github.com/catianag/GASTeNv2)
  * Luuk’s [GASTeN-HPO](https://github.com/luukgr/GASTeN-HPO)

The trained GAN **generates realistic boundary samples** that expose classifier blind-spots and often fool humans.

---

## 1 · Quick Start

```bash
# 1. Create & activate clean Python 3.10 environment
mamba create -n gasten python=3.10 && mamba activate gasten

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create minimal .env (adjust paths)
echo -e "CUDA_VISIBLE_DEVICES=0\nFILESDIR=/abs/path/for/large_files\nENTITY=my-wandb-team" > .env

# 4. Download a dataset (example: MNIST)
python -m src.data_loaders --download mnist --root $FILESDIR/data
```

---

## 2 · Preparation Pipeline

| Step | Purpose                     | Command                                                                                                               |
| ---- | --------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 1    | Compute FID reference stats | `python src/gen_pairwise_inception.py --dataset mnist`                                                                |
| 1′   | FID stats for a single pair | `python -m src.metrics.fid --data $FILESDIR/data --dataset mnist --pos 7 --neg 1`                                     |
| 2    | Train a classifier          | `python src/gen_classifiers.py --classifier-type cnn --pos 7 --neg 1 --epochs 50 --nf 50 --seed 4441 --dataset mnist` |
| 3    | Generate fixed noise pool   | `python src/gen_test_noise.py --nz 2048 --z-dim 64`                                                                   |

> **Tip:** To train multiple CNN widths, use e.g. `--nf 4-8,8-16-32,20`.

---

## 3 · Two-Stage HPO Training

| Stage      | Goal                                   | Entry-point                             | Example                                                                                                  |
| ---------- | -------------------------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Step-1** | Base GAN (opt. FID)                    | `gasten_bayesian_optimization_step1.py` | `python -m src.optimization.gasten_bayesian_optimization_step1 --config experiments/mnist-7v1_step1.yml` |
| **Step-2** | Fine-tune with Gaussian ambiguity loss | `gasten_bayesian_optimization_step2.py` | `python -m src.optimization.gasten_bayesian_optimization_step2 --config experiments/mnist-7v1_step2.yml` |

Each writes a JSON (best HPO) and TXT (metrics summary). Swap YAMLs to target different datasets or models.

---

## 4 · Supported Datasets

| Dataset         | Size      | Channels | Notes            |
| --------------- | --------- | -------- | ---------------- |
| MNIST           | 28 × 28   | 1        | baseline         |
| Fashion-MNIST   | 28 × 28   | 1        | harder textures  |
| CIFAR-10        | 32 × 32   | 3        | colour           |
| STL-10          | 96 × 96   | 3        | larger, few-shot |
| ChestXRay       | 128 × 128 | 1        | medical          |
| ImageNet subset | 224 × 224 | 3        | scalability demo |

---

## 5 · Generate Boundary Samples (Step-2)

```bash
python -m src.optimization.after_step2 \
  <run_dir>/40 \
  results/models/<clf_dir> \
  --dataset fashion-mnist \
  --num 10000 \
  --batch 64
```

This will:

1. Load GAN and classifier
2. Sample `--num` images from GAN
3. Filter by classifier margin `|p−0.5|<0.1`
4. Score FID & ambiguity on each batch
5. Arrange the resulting 10k images left→right by decreasing ambiguity and save as a grid

---

## 6 · Explainability Pipeline (xAI)

```bash
python -m src.optimization.step2_xai \
  --sprite samples/grid.png \
  --cell-h 28 --cell-w 28 \
  --cols 100 \
  --n-samples 200 \
  --pick middle \
  --clf-checkpoint results/models/<clf_dir> \
  --dataset fashion-mnist \
  --split test \
  --gmm-clusters 8 \
  --batch-size 128 \
  --device cuda \
  --out-dir xai_outputs
```

This will:

1. Slice the grid into `cell-h×cell-w` tiles
2. Keep `|p−0.5|<0.1`, select `--n-samples`
3. Compute UMAP embeddings + GMM clustering (silhouette & DB)
4. Identify cluster medoids
5. Generate per-medoid GradientShap heatmaps

---

## 7 · Metrics & Outputs

* **FID**: Fréchet Inception Distance
* **ConfDist**: classifier ambiguity distance
* **Hubris**: overconfidence score
* **Precision/Recall**, **histograms**
* **UMAP + GMM** visualizations + **Heatmaps**

---

## 8 · License & Credits

| Item                | License / Source                                          |
| ------------------- | --------------------------------------------------------- |
| Code                | CC BY-NC-ND 4.0 (non-commercial research/teaching only)   |
| Original GASTeN     | [luispcunha/GASTeN](https://github.com/luispcunha/GASTeN) |
| Gaussian-loss ideas | [catianag/GASTeNv2](https://github.com/catianag/GASTeNv2) |
| SMAC workflow       | [luukgr/GASTeN-HPO](https://github.com/luukgr/GASTeN-HPO) |