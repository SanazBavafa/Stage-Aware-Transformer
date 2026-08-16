## Published Paper

This repository contains the implementation accompanying the following published paper:

**Stage-Aware Transformer for Patient Trajectory Modeling in ICU Mortality Prediction**

**Authors:** Sanaz Bavafa, Glory Ojoma Simon, and Guojun Liang

**Published in:** 2026 2nd International Conference on Federated Learning and Intelligent Computing Systems (FLICS)

**DOI:** [10.1109/FLICS70075.2026.11621949](https://doi.org/10.1109/FLICS70075.2026.11621949)

The proposed model achieves an **AUROC of 0.913**, **AUPRC of 0.391**, and **Min(Se, P+) of 0.401** on the MIMIC-III benchmark for in-hospital mortality prediction. The model improves AUPRC by **0.068** compared with the StageNet baseline.

For the complete methodology, experimental setup, results, and discussion, please refer to the published paper.


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
- `data/`: Directory for dataset (train, test)
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

## Data Preparation

The MIMIC-III dataset is not included in this repository. To use this project, please obtain the data from [MIMIC-III PhysioNet](https://physionet.org/content/mimiciii/1.4/) and download the required CSV files.

For the decompensation prediction task, build the benchmark dataset using the instructions provided in [mimic3-benchmarks](https://github.com/YerevaNN/mimic3-benchmarks/).

Once the benchmark is prepared, place the files from the `decompensation` directory into the `data/` directory. Sample files are included in the repository to demonstrate the expected input format and directory structure.

## Test Stage-Aware Transformer with MIMIC-III without training

The pre-trained weights provided in this repository (./saved_weights/SAT) can be used to reproduce the reported performance in the published paper.

You need to run train.py in test mode and input the data directory. For example,

$ python train.py --test_mode=1 --file_name SNK

## Training Stage-Aware Transformer

You need to have the dataset directory and file name to save model. For example,

$ python train.py --data_path='./data/' --file_name='trained_model' 

## Test the trained model
To evaluate a saved model checkpoint in test-only mode, specify the same file name that was used during training:
```bash
python train.py --test_mode 1 --file_name SAT
```

## Citation

If you use this repository or the Stage-Aware Transformer model in your research, please cite our paper:

```bibtex
@inproceedings{bavafa2026stage,
  title={Stage-Aware Transformer for Patient Trajectory Modeling in ICU Mortality Prediction},
  author={Bavafa, Sanaz and Simon, Glory Ojoma and Liang, Guojun},
  booktitle={2026 2nd International Conference on Federated Learning and Intelligent Computing Systems (FLICS)},
  pages={545--552},
  year={2026},
  organization={IEEE},
  doi={10.1109/FLICS70075.2026.11621949}
}
```