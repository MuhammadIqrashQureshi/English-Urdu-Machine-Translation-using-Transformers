# English-Urdu-Machine-Translation-using-Transformers
This project implements a Transformer-based Neural Machine Translation (NMT) system for translating English text into Urdu. The model fine-tunes Facebook’s mBART-50 (Many-to-Many Multilingual Model) using a parallel English-Urdu corpus.

This repository covers:
Dataset preparation
Tokenization
Model training
Evaluation using BLEU
Logging and saving predictions
Final model export

# Project Objectives
The goal of this task was to design and train a high-quality English→Urdu translation system using Transformer architectures.

# Primary Objectives
Implement a transformer-based NMT model (mBART-50).
Train the model on a suitable English–Urdu parallel corpus.
Evaluate translation quality using BLEU score.
Provide sample translations and qualitative analysis.

# Datasets Used
# Parallel Corpus for English-Urdu Language (Kaggle)
Contains 24,000+ sentence pairs
Source: Quran, dialogues, common phrases
Good for general-purpose translation
Kaggle link (download via API)

# Dataset Split
Train:	19,864
Validation:	2,208
Test:	2,453

# Model Architecture
mBART-50 (Many-to-Many Multilingual Translation Model)
610M parameters
Fully transformer-based encoder–decoder architecture
Trained on 50 languages
Supports both pretraining and multilingual fine-tuning

# Why mBART for English→Urdu?
Strong performance on low-resource languages
Pre-trained on monolingual + multilingual corpora
Allows fine-tuning with relatively small datasets
Good tokenization for Urdu via SentencePiece

# Implementation Details
Libraries Used
Hugging Face Transformers
Hugging Face Datasets
PyTorch
SentencePiece
SacreBLEU
Accelerate

# Major Steps in the Pipeline
1. Kaggle API Configuration & Dataset Download
2. Dataset Loading
3. Train/Validation/Test Split
4. Tokenization
5. Preprocessing & Encoding
6. Fine-Tuning with Seq2SeqTrainer
7. Evaluation

# Results
BLEU: 58.90
<img width="317" height="156" alt="image" src="https://github.com/user-attachments/assets/80e670b7-f97f-46ea-b90a-c78bffc5b470" />

# Project Structure
├── data/
│   ├── english-corpus.txt
│   ├── urdu-corpus.txt
├── final_model_en_ur_mbart/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   └── vocab files
├── translations.csv
├── experiment_log_.txt
├── Model Implementation.py
└── README.md

# How to Run the Code
1. Install Dependencies
   pip install transformers datasets sacrebleu sentencepiece evaluate kaggle torch accelerate
2. Configure Kaggle API
   Place your kaggle.json in:
   ~/.kaggle/
3. Run Training
   python model-implementation.py
4. Evaluate Model
   Automatically runs after training.

# Conclusion
This project demonstrates the full pipeline for building a modern translation system using Transformer models and multilingual pretrained architectures.
With a BLEU score of 58.9, the system achieves strong translation quality for Urdu, a traditionally low-resource language.
