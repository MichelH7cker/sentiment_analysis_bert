# Sentiment Analysis with BERT

This repository contains code for a sentiment analysis project using BERT (Bidirectional Encoder Representations from Transformers). The implementation leverages pre-trained BERT models to classify text into sentiment categories such as positive, negative, or neutral.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Predict](#predict)
- [License](#license)

## Features
- Preprocessing pipeline for text data.
- Fine-tuning of pre-trained BERT models for sentiment analysis.
- Supports custom datasets.

## Requirements
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Transformers (Hugging Face library)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib (optional, for visualizations)

Install the dependencies using the following command:
```bash
pip install -r requirements.txt
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/MichelH7cker/sentiment_analysis_bert.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment_analysis_bert
   ```
3. Install the required dependencies as mentioned above.

## Usage

### Data Preparation
- Place your dataset in the `dataset/` directory.
- The code currently uses a model that I have provided on the Hugging Face Hub, so some modifications may be necessary to ensure it runs correctly.

### Training
Run the training script:
```bash
python train.py
```

### Prediction
Make predictions on new text:
```bash
python predict.py "Your text here"
```

## Training
- Modify the configuration in `train.py` to change hyperparameters such as learning rate, batch size, etc.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to open an issue or contribute to the project if you have suggestions or improvements!
