# 🎬 Neural Collaborative Filtering (NCF) Recommender System

This repository demonstrates how to build a basic Deep Learning-based Collaborative Filtering Recommender System using Keras and TensorFlow. The model learns latent representations (embeddings) for users and items to predict user ratings.

## 🚀 Model Architecture

1. Input Layers: Accepts individual user and item identifiers as discrete integer inputs.
2. Embedding Layers: Maps sparse user and item IDs into dense latent vectors of specified dimensionality (`embedding_dim = 8`).
3. Flatten & Dot Product: Flattens the embedding tensors and calculates the dot product between the user vector and item vector to capture interaction strength.
4. Output Layer: Passes the dot product through a Dense layer with a single unit to output continuous predicted ratings.
5. Optimization: Uses the Adam optimizer and Mean Squared Error (MSE) loss function for model training.

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Scikit-Learn

## 💻 Installation & Usage

1. Install dependencies:
pip install tensorflow numpy scikit-learn

2. Run the script:
python recommender_ncf.py

## 📊 Evaluation & Inference

- Evaluates test loss (MSE) on unseen user-item interaction pairs.
- Generates rating predictions for specific user-item queries (e.g., predicting the rating for `User 0` and `Item 0`).
