# 🛠️ tiny-transformer-tr
*Build, train & deploy a **small-footprint Turkish GPT** model entirely from first-principles. 100 % reproducible on Google Colab or any CUDA‑enabled Windows workstation.*

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GH_USER>/tiny-transformer-tr/blob/main/notebook.ipynb)  
[![🤗 Model Card](https://img.shields.io/badge/HF%20Model-tiny--transformer--tr-orange?logo=huggingface)](https://huggingface.co/<YOUR_HF_USER>/tiny-transformer-tr)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![CI](https://github.com/<YOUR_GH_USER>/tiny-transformer-tr/actions/workflows/ci.yml/badge.svg)](https://github.com/<YOUR_GH_USER>/tiny-transformer-tr/actions)

---

## 0. Why does this repo exist?
> *Because technical interviewers worldwide value engineers who can **explain a transformer layer by layer** — not just call `from_pretrained()`.*

* **Engineering challenge** – we build every stage (tokenizer → data loader → model → evaluation) from scratch so you can answer deep‑dive questions.  
* **Scientific rigor** – experiments logged with *Weights & Biases*; scripts deterministic; seeds pinned.  
* **Production mindset** – ready for HF Hub, Docker, ONNX & Triton inference.  

---

## 1 · TL;DR
| Item | Spec |
|------|------|
| Corpus | Turkish Wikipedia 2022‑03 (≈ 2 GB after cleaning) |
| Tokenizer | Byte‑Level BPE · 30 k vocab |
| Model | GPT‑2‑mini 6 L × 8 H × 256 D (≈ 30 M params) |
| Training | 3 epochs · batch size 16 (grad accum 4) · fp16 |
| Cost | ≈ 2 GPU‑hours (Tesla T4) |
| Metrics | loss ≈ 2.8 · ppl ≈ 16.7 |

---

## 2 · Quick Start
```bash
# clone
git clone https://github.com/<YOUR_GH_USER>/tiny-transformer-tr.git && cd tiny-transformer-tr

# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train (local CUDA)
python train.py --epochs 3 --batch 4 --mixed_precision fp16
```
Or hit the **Colab** badge above and run every cell – free GPU, zero setup.

---

## 3 · Data Pipeline
```
Wikipedia dump → clean_text.py → data/wiki_tr.txt (2 GB)
```
Cleaning rules:
* Strip HTML, tables, refs  
* Turkish sentences only (`langdetect` ≥ 0.9)  
* Collapse multiple spaces → one space  

---

## 4 · Tokenizer
`tokenizers` **ByteLevelBPETokenizer** is trained **from scratch**:
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["data/wiki_tr.txt"],
    vocab_size=30_000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)
```
Artifacts live in `tokenizer_tr/` and can be pushed to the Hub so the community can reuse them.

---

## 5 · Model Architecture
```
6 × [Multi‑Head Attention 8 H → GELU → Residual + LayerNorm]
     ↳ hidden size 256 · bias False · dropout 0.1
LM head ties weights with input embeddings.
```
Design decisions:
* **Context 512 tokens** – fits into 14 GB RAM with fp16.  
* **Weight tying** – parameter efficient, faster convergence.  
* **No activation checkpointing** – small model, not worth overhead.  

---

## 6 · Training Strategy
* **Optimizer** – AdamW β=(0.9, 0.95) · ε=1e‑8  
* **LR Scheduler** – Linear warm‑up 5 % → cosine decay  
* **Gradient Accumulation** – 4 steps (effective batch 16) to squeeze into T4 GPU memory.  
* **Mixed Precision** – Automatic fp16 with PyTorch `amp`.  
* **Checkpointing** – every 500 steps; keep 2 latest.  
* **Logging** – TensorBoard + W&B (optional).  

> **Compute budget:** 2 h (T4) or 45 min (A100) ≈ $0 (Colab) / $0.5 (spot A100).

---

## 7 · Evaluation
* Perplexity on held‑out 50 k sentences.  
* `evaluate.py` script provides BLEU & perplexity.  
* Qualitative generation samples stored under `samples/`.  

---

## 8 · Deployment
* **Push to HF** – `python push_to_hub.py`  
* **Gradio Space** – ready‑made `app.py` gives chat UI.  
* **ONNX** – `python export_onnx.py` converts & quantizes → 2× faster CPU inference.  
* **Docker** – `docker build -t ttt-infer -f docker/Dockerfile .`  

---

## 9 · Results & Samples
```text
> Merhaba, bugün hava nasıl?
>>>> Bugün İstanbul genelinde parçalı bulutlu bir gökyüzü bekleniyor. Sıcaklık öğleden sonra 24–26 °C aralığında…
```
More examples in `samples/`.

---

## 10 · Project Structure
```text
├── data/                  # raw + cleaned corpora
├── tokenizer_tr/          # BPE vocab/merges
├── tiny-transformer-tr/   # final model snapshot
├── scripts/
│   ├── clean_text.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   └── push_to_hub.py
├── train.py               # main training entry
├── requirements.txt
└── README.md
```

---

## 11 · Contributing
PR’ler, issue’lar & feature önerileri memnuniyetle karşılanır. Lütfen `pre-commit` hook’larını çalıştırın:
```bash
pre-commit install
pre-commit run --all-files
```

---

## 12 · License
MIT – özgürce kullan, yıldız bırakırsan ⭐ mutlu oluruz.

---

> © 2025 <Deniz_Sakaroglu> · Made in Türkiye with 🧠 & ☕
