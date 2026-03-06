<div align="center">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=32&duration=3500&pause=800&color=58A6FF&center=true&vCenter=true&multiline=true&repeat=true&width=800&height=120&lines=Neural+Architectures+from+Scratch;LLMs+·+VLMs+·+Diffusion+·+Flow+Models;Built+from+First+Principles+in+PyTorch)](https://git.io/typing-svg)

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-58A6FF?style=for-the-badge)](./LICENSE)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Language](https://img.shields.io/badge/Language-Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Actively_Growing-00C853?style=for-the-badge)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-FF6B6B?style=for-the-badge)]()

<br/>

> **"The only way to truly understand a model is to build it from scratch — the math, the gradients, and every tensor operation."**

</div>

---

## 📌 What Is This Repository?

This is a **single, growing reference repository** of foundational AI architectures implemented entirely from scratch in PyTorch — no `transformers`, no high-level wrappers, no black boxes.

Every model here starts from raw tensor operations and builds up to the full architecture. The implementations are intentionally **minimal, readable, and mathematically faithful** to the original papers. This repository covers:

- 🖼️ **Vision models** — ViT, and more
- 🎨 **Generative & Diffusion models** — DiT, DDPM, Score Models, Flow Matching
- 🧠 **Language models** — Decoder-only Transformers, LLMs
- 🌐 **Multimodal models** — VLMs, Multimodal LLMs
- 🌊 **Continuous generative models** — Neural ODEs, SDEs, Flow Matching

If you want to understand *why* these architectures work — not just *how* to use them — this is the right place.

---

## 🗂️ Table of Contents

- [Implemented Architectures](#-implemented-architectures)
- [Repository Structure](#-repository-structure)
- [Quickstart](#-quickstart)
- [Model Details](#-model-details)
  - [ViT — Vision Transformer](#-vit--vision-transformer)
  - [DiT — Diffusion Transformer](#-dit--diffusion-transformer)
- [Roadmap](#-roadmap)
- [Design Philosophy](#-design-philosophy)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✅ Implemented Architectures

<div align="center">

| # | Model | Domain | Paper | Status |
|:-:|:------|:-------|:------|:------:|
| 01 | **ViT** — Vision Transformer | Computer Vision | [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929) | ✅ Done |
| 02 | **DiT** — Diffusion Transformer | Generative AI | [Peebles & Xie, 2022](https://arxiv.org/abs/2212.09748) | ✅ Done |
| 03 | **GPT** — Decoder-only Transformer | Language Models | [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762) | 🔜 Soon |
| 04 | **DDPM** — Denoising Diffusion | Generative AI | [Ho et al., 2020](https://arxiv.org/abs/2006.11239) | 🔜 Soon |
| 05 | **Score Matching** | Generative AI | [Song & Ermon, 2019](https://arxiv.org/abs/1907.05600) | 📋 Planned |
| 06 | **Flow Matching** | Generative AI | [Lipman et al., 2022](https://arxiv.org/abs/2210.02747) | 📋 Planned |
| 07 | **VLM** — Vision-Language Model | Multimodal | — | 📋 Planned |
| 08 | **Multimodal LLM** | Multimodal | — | 📋 Planned |

</div>

---

## 📂 Repository Structure

Each architecture lives in its own self-contained directory with a consistent, modular layout:

```
neural-architectures-from-scratch/
│
├── ViT/                          # Vision Transformer
│   ├── vit.py                    #   ← Main model: patch embed + [CLS] + Transformer
│   ├── attention.py              #   ← Multi-head self-attention (from scratch)
│   ├── transformer_layer.py      #   ← Pre-norm encoder block (MHA + FFN)
│   ├── tools.py                  #   ← Patch extraction, positional encoding
│   ├── config.yaml               #   ← All hyperparameters in one place
│   └── __init__.py
│
├── DiT/                          # Diffusion Transformer
│   ├── dit.py                    #   ← Main model: timestep & label conditioning + adaLN
│   ├── attention.py              #   ← Multi-head self-attention (from scratch)
│   ├── transformer_layer.py      #   ← Pre-norm DiT block with adaLN-Zero
│   ├── tools.py                  #   ← 2D sinusoidal encoding, timestep MLP, patch ops
│   ├── config.yaml               #   ← All hyperparameters in one place
│   └── __init__.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

> **Design Rule:** Every model directory is fully self-contained. You can copy any single folder into another project and it will work independently.

---

## ⚡ Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/asifcsai/neural-architectures-from-scratch.git
cd neural-architectures-from-scratch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` contains only:
```
torch>=2.0.0
pyyaml
```

### 3. Run any model

```bash
# Run Vision Transformer
python ViT/vit.py

# Run Diffusion Transformer
python DiT/dit.py
```

Both scripts will print the model config and verify the forward pass with a dummy batch:

```
{'img_height': 224, 'img_width': 224, 'patch_height': 16, 'patch_width': 16,
 'd_model': 768, 'num_heads': 12, 'num_layers': 12, ...}
Input shape:  torch.Size([2, 3, 224, 224])
Output shape: torch.Size([2, 1000])       # ViT
Output shape: torch.Size([2, 3, 224, 224]) # DiT
```

---

## 🔍 Model Details

### 🖼️ ViT — Vision Transformer

> Paper: [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al., 2020

**Core idea:** Treat an image as a sequence of fixed-size patches and process them with a standard Transformer encoder — no convolutions, no inductive bias.

**Architecture pipeline:**

```
Image [B, 3, H, W]
    │
    ▼
Patch Extraction → [B, N, patch_h × patch_w × 3]   (N = number of patches)
    │
    ▼
Linear Projection → [B, N, d_model]                 (patch embedding)
    │
    ▼
Prepend [CLS] Token → [B, N+1, d_model]
    │
    ▼
Add Learnable Positional Embeddings → [B, N+1, d_model]
    │
    ▼
L × Transformer Encoder Blocks
│   ├── LayerNorm (Pre-Norm)
│   ├── Multi-Head Self-Attention
│   ├── Residual Connection
│   ├── LayerNorm
│   ├── MLP (FC → GELU → FC)
│   └── Residual Connection
    │
    ▼
Extract [CLS] Token → [B, d_model]
    │
    ▼
Classification Head (Linear) → [B, num_classes]
```

**Default config (`ViT/config.yaml`):**

```yaml
img_height: 224
img_width: 224
patch_height: 16
patch_width: 16
d_model: 768
num_heads: 12
num_layers: 12
mlp_ratio: 4.0
dropout: 0.1
batch_size: 2
```

**Usage:**

```python
import torch, yaml
from ViT.vit import ViT

with open('ViT/config.yaml') as f:
    config = yaml.safe_load(f)

model = ViT(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

x = torch.randn(2, 3, 224, 224).to(device)
logits = model(x)
print(logits.shape)  # torch.Size([2, 1000])
```

---

### 🎨 DiT — Diffusion Transformer

> Paper: [*Scalable Diffusion Models with Transformers*](https://arxiv.org/abs/2212.09748) — Peebles & Xie, 2022

**Core idea:** Replace the U-Net backbone in diffusion models entirely with a Transformer. Condition each block on the diffusion timestep `t` (and optionally class label `y`) using **Adaptive Layer Norm (adaLN-Zero)** — the parameters of LayerNorm are predicted dynamically from the conditioning signal, not learned as static weights.

**Architecture pipeline:**

```
Noisy Image x_t [B, C, H, W]    Timestep t [B]
    │                                 │
    ▼                                 ▼
Patch Extraction                  Sinusoidal Embedding
    │                             → MLP → [B, d_model]
    ▼                                 │
Linear Projection → [B, N, d_model]   │
    │                                 │
    ▼                                 ▼
Add 2D Sinusoidal Pos Encoding        Conditioning Vector c
    │                                 │
    └──────────────┬───────────────────┘
                   │
                   ▼
        L × DiT Transformer Blocks
        │
        │   ┌─ adaLN-Zero ─────────────────────────────────────┐
        │   │  c → MLP → (γ₁, β₁, α₁, γ₂, β₂, α₂)            │
        │   │  LayerNorm(x) * (1 + γ) + β  (modulated norm)    │
        │   │  α controls residual gate (init α=0 → identity)  │
        │   └───────────────────────────────────────────────────┘
        │   ├── adaLN → Multi-Head Self-Attention → α₁ · attn + x
        │   └── adaLN → MLP (FC → GELU → FC)     → α₂ · ffn  + x
                   │
                   ▼
        Final LayerNorm + Linear Unpatch
                   │
                   ▼
        Predicted Noise ε_θ(x_t, t) [B, C, H, W]
```

**Default config (`DiT/config.yaml`):**

```yaml
img_height: 224
img_width: 224
patch_height: 16
patch_width: 16
d_model: 768
num_heads: 12
num_layers: 12
mlp_ratio: 4.0
dropout: 0.1
time_emb_dim: 128
batch_size: 2
```

**Usage:**

```python
import torch, yaml
from DiT.dit import DiT

with open('DiT/config.yaml') as f:
    config = yaml.safe_load(f)

model = DiT(config)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

x_t = torch.randn(2, 3, 224, 224).to(device)   # noisy image
t   = torch.randint(0, 1000, (2,)).to(device)   # timestep

predicted_noise = model(x_t, t)
print(predicted_noise.shape)  # torch.Size([2, 3, 224, 224])
```

---

## 🗺️ Roadmap

```
Phase 1 — Vision & Generative Transformers       ████████████████  DONE
  ✅ Vision Transformer (ViT)
  ✅ Diffusion Transformer (DiT)

Phase 2 — Language Models                        ████░░░░░░░░░░░░  IN PROGRESS
  🔜 Decoder-only Transformer (GPT-style)
  🔜 Byte-Pair Encoding (BPE Tokenizer)
  🔜 Rotary Positional Embeddings (RoPE)

Phase 3 — Diffusion & Score Models               ░░░░░░░░░░░░░░░░  PLANNED
  📋 DDPM — Denoising Diffusion Probabilistic Models
  📋 DDIM — Accelerated Sampling
  📋 Score Matching (Song & Ermon)
  📋 SDE-based Generative Models

Phase 4 — Continuous Generative Models           ░░░░░░░░░░░░░░░░  PLANNED
  📋 Neural ODEs
  📋 Continuous Normalizing Flows (CNF)
  📋 Flow Matching (Lipman et al.)
  📋 Rectified Flow

Phase 5 — Multimodal Systems                     ░░░░░░░░░░░░░░░░  PLANNED
  📋 Vision-Language Model (VLM)
  📋 Multimodal LLM (vision encoder + LLM backbone)
  📋 Cross-modal Attention
```

---

## 💡 Design Philosophy

This repository follows three strict rules in every implementation:

**1. Start from tensors, not abstractions.**
No `nn.TransformerEncoder`, no `AutoModel.from_pretrained`. Every attention head, every norm layer, every positional encoding is written from scratch. If it's in the paper, it's in the code.

**2. Readability over speed.**
Variable names mirror the paper notation. Comments explain *why* an operation is done, not just *what* it does. The goal is that someone reading the code alongside the paper can follow both simultaneously.

**3. One config file per model.**
All hyperparameters — dimensions, heads, layers, dropout — live in a single `config.yaml`. Switching from ViT-Base to ViT-Large is a one-line config change.

---

## 🤝 Contributing

Contributions are welcome — especially new from-scratch implementations or mathematical annotations.

```bash
# 1. Fork the repository
# 2. Create a feature branch
git checkout -b feat/add-ddpm

# 3. Follow the existing module structure:
#    new_model/
#      ├── model.py
#      ├── attention.py       (if applicable)
#      ├── transformer_layer.py (if applicable)
#      ├── tools.py
#      ├── config.yaml
#      └── __init__.py

# 4. Verify forward pass with a dummy batch before opening a PR
# 5. Submit a pull request with a brief description of the architecture
```

Please keep implementations **minimal and educational**. Optimization tricks (flash attention, fused kernels, etc.) belong in a separate branch or note — the main branch should always be the clearest version.

---

## 👤 Author

<div align="center">

**Asif Miah**
AI Researcher · NSU Machine Intelligence Lab · Dhaka, Bangladesh

[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--2465--8056-A6CE39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0001-2465-8056)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-asif--miah-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/asif-miah-608ba9256)
[![Email](https://img.shields.io/badge/Email-asif.cs.ai@gmail.com-EA4335?style=flat-square&logo=gmail&logoColor=white)](mailto:asif.cs.ai@gmail.com)
[![YouTube](https://img.shields.io/badge/YouTube-NeuralCraft-FF0000?style=flat-square&logo=youtube&logoColor=white)](https://www.youtube.com/channel/UCM3aSkutbQADf1bRjjOHv7g)

*Research interests: Generative models · Flow Matching · Score-based models · Multimodal VLMs*

</div>

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE) — free to use, study, modify, and build on.

---

<div align="center">

**If this repository helped you understand how these architectures actually work, consider leaving a ⭐**

*More architectures coming soon — watch/star to stay updated.*

</div>
