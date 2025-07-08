# Amharic Text Generation using GPT & PyTorch
A custom-built Amharic language model powered by the GPT architecture and implemented in PyTorch. This project demonstrates how transformer-based models can be adapted for underrepresented languages like Amharic, enabling token-level text generation with a focus on linguistic relevance and model convergence.

Overview
This model is trained on raw Amharic text (am_data.txt) and is capable of generating coherent Amharic sentences by predicting the next token in a given sequence. Designed from scratch, this project is an educational deep dive into building a transformer-based language model without relying on external libraries like HuggingFace Transformers.

Features
 Custom GPT Architecture
Built with transformer layers and multi-head self-attention mechanisms.

Amharic Text Dataset
Trained on native Amharic data from am_data.txt.

Token-Level Text Generation
Predicts the next token in a sequence to generate fluent Amharic text.

Configurable Hyperparameters
Easily tune model depth, embedding size, learning rate, etc., via script.

Train/Validation Support
Supports dataset splitting for training and evaluation.

Optimized Weight Initialization
Incorporates proven initialization strategies to improve convergence.

Tech Stack
Language: Python (PyTorch)

Model: GPT-inspired Transformer

Tokenizer: Custom, character/token-based

Training: Script-driven with CLI configuration
