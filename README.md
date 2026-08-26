# LLM Classification (BERT and Transformer Classification Projects)

# AG News Classification with BERT & Transformer Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)

## Overview

This repository is text classification on the **AG News dataset** using three different approaches:

1. **BERT Fine-tuning** with a custom classifier head (Feature Extraction)
2. **Transformer from Scratch** (Custom Transformer implementation)
3. **GPT-2 Generated Data Augmentation** for training BERT on synthetic data

The project demonstrates text classification tasks.

## Project Structure

```
.
├── bert_runner.py                          # BERT feature extraction training
├── Transformer_runner.py                   # Transformer training
├── GPT_bert_runner.py                      # GPT-2 data generation + BERT training
├── My_BertClassifier.py                    # BERT classifier wrapper class
├── My_Transformer.py                       # Transformer from scratch implementation
├── GPT2_text_generator.py                  # GPT-2 text generation module
├── bert_classifier.pth                     # Trained BERT classifier weights
├── Transformer_classifier.pth              # Trained Transformer weights
├── bert_classifier_training_curves.png
├── Transformer_classifier_training_curves.png
├── GPT_bert_classifier_training_curves.png
├── torch weights are avalavbe request to me MasoudMahan1997@gmail.com
├── GPT-DATA generated and Data are avalavbe request to me MasoudMahan1997@gmail.com  ᶻ 𝗓 𐰁 .ᐟ
```

## Models

### 1. BERT Classifier (`bert_runner.py`, `My_BertClassifier.py`)

Uses pre-trained **BERT-base-uncased** as a feature extractor with a custom classification head:

```
BERT-base → Freeze → Custom MLP Head → Output
```

**Architecture:**
- **Base**: `bert-base-uncased` (110M parameters)
- **Classifier**: 2-layer MLP (768 → 128 → 4)


### 2. Custom Transformer (`Transformer_runner.py`, `My_Transformer.py`)

A **Transformer encoder** implemented from scratch:

```
Input → Token Embedding → Positional Encoding → 
TransformerBlock × depth → Global Average Pooling → 
Classification Head → Log-Softmax
```

**Architecture:**
- **Embedding**: 128-dim token embeddings
- **Attention**: 8-head self-attention
- **Depth**: 8 transformer blocks
- **Positional**: Sinusoidal positional encoding
- **Feed-Forward**: MLP with ReLU (128 → 256 → 128)
- **Parameters**: ~1.5M trainable parameters

### 3. GPT-2 Augmented Training (`GPT_bert_runner.py`)

Generates synthetic training data using **GPT-2** and trains a BERT classifier:

1. Sample 50,000 examples from AG News training set
2. Use GPT-2 to generate text continuations
3. Train BERT on generated texts
4. Evaluate on original test set

## Installation

```bash
# Clone the repository
git clone https://github.com/MasoudMahanian/LLM-Classification.git

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets pandas numpy tqdm matplotlib
pip install scikit-learn  # for evaluation metrics
```

## Usage

### Train BERT Classifier
```bash
python a1.py
```
- Loads AG News dataset
- Freezes BERT and trains custom classifier
- Saves model to `bert_classifier.pth`
- Generates training curves

### Train Custom Transformer
```bash
python a2.py
```
- Trains transformer from scratch on AG News
- Uses combined title + description
- Saves model to `Transformer_classifier.pth`

### GPT-2 Data Augmentation
```bash
python ss.py
```
- Generates synthetic text with GPT-2
- Trains BERT on generated data
- Saves model to `GPT_bert_classifier.pth`

### Use Models for Inference

```python
from My_BertClassifier import BertClassifier

classes = {0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
classifier = BertClassifier(classes=classes)

# Load trained model
classifier.model_bert.load_state_dict(torch.load('bert_classifier.pth'))

```

## Results

| Model | Train Accuracy | Test Accuracy | Parameters |
|-------|---------------|---------------|------------|
| BERT Classifier | ~92% | ~91% | 110M |
| Custom Transformer | ~85% | ~83% | ~1.5M |
| GPT-2 Augmented BERT | ~89% | ~87% | 110M |

### Training Curves

<p align="center"> <img src="https://github.com/MasoudMahanian/LLM-Classification/blob/main/Result/bert_classifier_training_curves.png" width="45%" /> <img src="https://github.com/MasoudMahanian/LLM-Classification/blob/main/Result/GPT_bert_classifier_training_curves.png" width="45%" /> </p><p align="center"><em>training Curves</em></p>



## Experiments

### Feature Extraction vs Fine-tuning
The BERT model uses feature extraction (freezing the base) to demonstrate transfer learning efficiency with limited computational resources.

### Transformer from Scratch
Custom implementation shows understanding of:
- Multi-head self-attention
- Positional encodings
- Layer normalization
- Residual connections

### Data Augmentation with GPT-2
Demonstrates how synthetic data can be used for:
- Improving model robustness
- Handling class imbalance
- Generating additional training examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AG News dataset](https://huggingface.co/datasets/sh0416/ag_news) from HuggingFace
- [BERT](https://huggingface.co/bert-base-uncased) from Google
- [GPT-2](https://huggingface.co/gpt2) from OpenAI
- PyTorch and HuggingFace Transformers libraries

## Contact

- GitHub: [@MasoudMahanian](https://github.com/MasoudMahanian)
- Email: MasoudMahan1997@gmail.com

---

<div align="center">
  PyTorch and Transformers
</div>
