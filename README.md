# Paddy Disease Classifier

An image classification notebook for identifying paddy (rice) leaf diseases with PyTorch and transfer learning.

The project uses a pretrained ResNet-18 model, adapts its classifier for the ten classes in the Kaggle Paddy Disease Classification dataset, and evaluates the model on a held-out validation split.

## What this project does

The notebook contains an end-to-end computer vision workflow:

1. Downloads the dataset from Kaggle.
2. Applies resizing, augmentation, and normalization.
3. Splits the training images into training and validation sets.
4. Fine-tunes a pretrained ResNet-18 model.
5. Tracks training loss, validation loss, and validation accuracy.

## Model and training details

- **Backbone:** ImageNet-pretrained ResNet-18
- **Classifier:** Dropout followed by a linear layer with 10 outputs
- **Input size:** 128 x 128 pixels
- **Dataset split:** 80% training and 20% validation
- **Loss:** Cross-entropy loss
- **Optimizer:** Adam
- **Training duration:** 50 epochs
- **Device:** CUDA GPU when available, otherwise CPU

## Dataset

The notebook uses the [Paddy Disease Classification dataset on Kaggle](https://www.kaggle.com/competitions/paddy-disease-classification). It contains ten classes: nine paddy leaf disease categories and one healthy category.

The dataset is downloaded at runtime, so it is not included in this repository.

## Repository contents

```text
.
|-- Disease_Classifier.ipynb  # Training and evaluation notebook
`-- README.md                 # Project documentation
```

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or JupyterLab
- A Kaggle account and API credentials
- A CUDA-capable GPU is recommended but not required

Install the packages used by the notebook:

```bash
python -m pip install opendatasets torch torchvision matplotlib numpy tqdm opencv-python pillow
```

Depending on your PyTorch installation, you may need to install a CUDA-specific build by following the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Kaggle credentials

The notebook uses `opendatasets` to download the competition data. When prompted, provide your Kaggle username and API key. You can create or download the key from your [Kaggle account settings](https://www.kaggle.com/settings/account).

Do not commit `kaggle.json` or any API credentials to this repository.

## Run the notebook

Clone the repository and open the notebook:

```bash
git clone https://github.com/upendra9798/Paddy-disease-classifier.git
cd Paddy-disease-classifier
jupyter notebook Disease_Classifier.ipynb
```

Run the cells from top to bottom. The notebook downloads the dataset, prepares the data loaders, trains the model, and displays evaluation metrics and visualizations.

For a faster interactive experience, the notebook can also be opened in [Google Colab](https://colab.research.google.com/).

## Reported results

The training run documented in the original notebook reported:

- **Validation accuracy:** 96.73%
- **Validation loss:** 0.1895

Results can vary with the random split, library versions, hardware, and training configuration. Treat these figures as a reference rather than a guaranteed benchmark.

## Notes and limitations

- This is an image classification experiment, not a production diagnosis system.
- Predictions may be unreliable for blurry images, unusual lighting, unseen disease symptoms, or plant varieties outside the training data.
- The notebook currently focuses on training and evaluation; it does not provide a deployed web or mobile interface.

## License

No license is currently specified for this repository. Add a license before redistributing the code or trained model.
