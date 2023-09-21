# Pytorch CUB birds 200 Classification with GUI

## Dependencies

- PyCharm 2023.2.1
- Pytorch 2.0.0
- PyQt5
- Other common packages (numpy, PIL, matplotlib etc.)

## Overview

This project trains an EfficientNet-b0 model from scratch to classify 200 species of birds from the CUB-200 dataset. The model architecture and training code is in `train_model.ipynb`. The trained model parameters are saved to `Best_model.pth`. A GUI is provided via PyQt for easy use of the model to classify new images.

## Model Architecture

The model architecture uses EfficientNet-b0 as the base model. The model is trained from scratch on CUB-200 dataset in `train_model.ipynb`.

## Training

The EfficientNet-b0 model is trained on CUB-200 dataset without any pretrained weights. Training is done in `train_model.ipynb` with details on data preprocessing, hyperparameters, training loop etc.

The best model parameters are saved to `Best_model.pth`. This trained model is loaded in `ui.py` for the GUI.

## Usage

1. Clone this repo
2. Ensure PyQt and other dependencies are installed
3. The UI layout is defined in `CUB-birds-200-classification.ui` made with Qt Designer. This `.ui` file is converted to `CUB-birds-200-classification.py` using `pyuic`.
4. The main GUI code is in `ui.py` which handles the backend logic.
5. Run `python ui.py` to launch the GUI

## Evaluation

The file `eval.txt` records evaluation results on the test set, including:

- Overall accuracy
- Per-class Precision, Recall and F1-score
- Overall Precision, Recall and F1-score
- Detailed model architecture information

This allows inspection of model performance across different evaluation metrics, and drill-down into per-class performance.

## References

- CUB-200 dataset: https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images

