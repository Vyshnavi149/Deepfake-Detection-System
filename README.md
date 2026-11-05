# Deepfake Detection System

DeepTrace AI is a Streamlit application that classifies uploaded images as
**real** or **fake** using a TensorFlow/Keras model based on MobileNetV2
transfer learning.

## Features

- Upload JPG, JPEG, or PNG images
- Resize and preprocess images automatically
- Predict real/fake classification with confidence scores
- Display analysis history during the current session
- Train and evaluate the model with a grouped train/validation/test split

## Project structure

```text
.
├── app.py                     # Streamlit application
├── train.py                   # Model training and evaluation
├── preprocess.py              # Dataset preprocessing utilities
├── requirements.txt           # Python dependencies
└── models/
    └── deepfake_model.keras   # Model used by the application
```

The training dataset is intentionally not included in this repository. Place
it locally in the following structure when training:

```text
dataset/
├── real/
└── fake/
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is empty, install the packages used by the project:

```bash
pip install streamlit tensorflow numpy pillow scikit-learn
```

## Run the application

From the project directory:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, upload an image, and select
**Analyse Image**.

## Train the model

1. Add labelled images under `dataset/real/` and `dataset/fake/`.
2. Run:

   ```bash
   python train.py
   ```

The training script creates `models/best_model.keras` and then saves the final
model as `models/deepfake_model.keras`. The latter is loaded by `app.py`.

## Important notes

- This is an experimental image-classification tool, not a forensic or legal
  verification system.
- Predictions depend on the training data and may be incorrect for unfamiliar
  images.
- Do not upload private or sensitive images to an untrusted deployment.
- Datasets, virtual environments, Node.js dependencies, environment files,
  secrets, and generated files are excluded through `.gitignore`.

## License

Add the license that applies to your project before publishing the repository.