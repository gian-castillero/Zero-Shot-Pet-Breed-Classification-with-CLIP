# Zero-Shot Pet Breed Classification with CLIP

Applying OpenAI's CLIP multimodal vision-language model for zero-shot pet breed classification, with prompt engineering to push accuracy above 87%; no fine-tuning required.

## Overview

This project uses OpenAI's CLIP (`clip-vit-base-patch32`), a multimodal transformer trained to align image and text representations in a shared embedding space, to classify 37 breeds of cats and dogs from the Oxford-IIIT Pet dataset. Unlike a CNN trained specifically on this task, CLIP performs **zero-shot classification**: it is never fine-tuned on the pet dataset at all. Instead, classification is done by comparing an image's embedding against text embeddings of candidate class descriptions and selecting the closest match.

The project then explores **prompt engineering**, modifying the text inputs without changing the model, to improve classification accuracy from ~80% to above 87%.

## Model

**CLIP** (Contrastive Language-Image Pretraining) consists of two encoders:
- A **Vision Transformer (ViT)** that encodes images into embedding vectors
- A **Text Transformer** that encodes text strings into embedding vectors

Both encoders are trained jointly on 400 million image-text pairs via contrastive learning: matched pairs are pulled together in embedding space, unmatched pairs are pushed apart. At inference time, classification is done by computing the cosine similarity between an image embedding and the text embeddings for each candidate label — no task-specific training required.

## Dataset

[Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) test split: ~3,700 images across 37 breeds (12 cat breeds, 25 dog breeds). Labels are breed indices into the `class_names` list.

## Task 1: Baseline Zero-Shot Classification

Text inputs consist of bare class names (e.g. `"Abyssinian"`, `"Pug"`). The image embedding is compared against all 37 text embeddings; the label with the highest logit is the prediction.

**Baseline accuracy: ~80%**

## Task 2: Prompt Engineering

The text inputs are enriched with species information drawn from the `class_to_species` mapping:

```
"Picture of a pet with breed: {breed}, which is a type of {cat/dog}"
```

This additional context helps CLIP disambiguate ambiguous breed names and leverages its pretraining on natural descriptive text rather than bare taxonomic labels.

**Accuracy after prompt engineering: >87%**

## Why CLIP Over a CNN?

A CNN trained on this dataset would be limited to the 37 breeds it was trained on. CLIP offers three qualitative advantages:

1. **Open vocabulary:** Any text string is a valid class label; no retraining for new categories
2. **Multimodal understanding:** Predicts from arbitrary natural language descriptions rather than fixed integer labels
3. **Prompt-improvable:** Performance can be increased through better text formulation without touching model weights

**Potential applications:**
- **Visual product search:** A customer describes a product in natural language; CLIP retrieves the closest matching product images from a catalog
- **Medical imaging:** A clinician describes diagnostic findings in text; CLIP retrieves similar cases or assists classification without requiring a purpose-built labeled dataset for every condition

## Tech Stack

- Python 3
- PyTorch
- Hugging Face `transformers` (`CLIPModel`, `CLIPProcessor`)
- `torchvision` (Oxford-IIIT Pet dataset)
- Matplotlib

## How to Run

```bash
pip install torch torchvision transformers matplotlib jupyter
jupyter notebook clip-multimodal.ipynb
```

The CLIP model and Oxford-IIIT Pet dataset download automatically on first run. GPU acceleration is recommended for faster inference across the full dataset.
