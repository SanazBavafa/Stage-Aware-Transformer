# Stage-Aware Transformer for ICU Mortality Prediction

This repository contains the implementation of a Stage-Aware Transformer model for ICU mortality prediction on the MIMIC-III benchmark. The project includes a training script, a reusable PyTorch model definition, and a Jupyter notebook pipeline for step-by-step experimentation and evaluation.

## Overview

The proposed model combines:
- Linear input projection for clinical features and interval features.
- Sinusoidal positional encoding for temporal ordering.
- Transformer encoder with causal masking and padding masking.
- Temporal convolution for local temporal pattern extraction.
- Residual connection, layer normalization, and sigmoid-based prediction head.

The pipeline follows the same logic as the training script:
1. Load and preprocess MIMIC-III time-series data.
2. Discretize irregular measurements into fixed time steps.
3. Normalize continuous channels.
4. Generate training, validation, and test batches.
5. Train the model with masked binary cross-entropy loss.
6. Save the best checkpoint based on validation AUC-PRC.
7. Load the best checkpoint and evaluate on the test set.

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

You also need the preprocessing utilities used by the original MIMIC-III benchmark pipeline.

## File Structure

- `train.py`: Full training and evaluation script.
- `model.py`: Defines the `StageAwareTransformer` architecture.
- `stage_aware_transformer_pipeline.ipynb`: Notebook version of the pipeline for experimentation.
- `saved_weights/`: Directory for saved checkpoint and weights.

## Hyperparameters

Default settings used in the pipeline:

- `input_dim = 76`
- `rnn_dim = 384`
- `output_dim = 1`
- `K = 10`
- `chunk_level = 3`
- `num_heads = 8`
- `num_layers = 2`
- `ff_dim = 512`
- `dropout_rate = 0.5`
- `dropconnect_rate = 0.5`
- `dropres_rate = 0.3`
- `max_len = 512`

## Training

The model is trained with:
- Batch size: `128`
- Epochs: `50`
- Learning rate: `1e-3`

Validation performance is monitored using AUC-PRC, and the best checkpoint is saved automatically.

## Evaluation

During evaluation:
- The best checkpoint is loaded from `saved_weights/`.
- The test set is processed with the same discretizer and normalizer.
- Masked predictions are used to compute test loss and binary classification metrics.
- ROC curve and AUROC are reported for the final model.

## Test Stage-Aware Transformer with MIMIC-III without training
Trained weights are provided in ./saved_weights/SAT and you can obtain the reported performance in our paper by simply load the weights to the model.

You need to run train.py in test mode and input the data directory. For example,

$ python train.py --test_mode=1 --file_name SNK

## Training Stage-Aware Transformer

You need to have the dataset directory and file name to save model. For example,

$ python train.py --data_path='./data/' --file_name='trained_model' 

## Test the trained model
To evaluate a saved model checkpoint in test-only mode, specify the same file name that was used during training:
```bash
python train.py --test_mode 1 --file_name SNK
```