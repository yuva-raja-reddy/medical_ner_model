# Medical Named Entity Recognition (NER) Model 🏥

A specialized NER model trained on medical text data to identify and extract medical entities such as conditions, medications, and pathogens.

## Features

- Custom-trained spaCy NER model for medical text
- Interactive Gradio web interface
- Support for multiple entity types
- Real-time entity visualization
- Pre-trained on medical dataset

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yuva-raja-reddy/medical_ner_model.git
cd medical_ner_model
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage
1. Run the Gradio interface:
```bash
python app.py
```
Access the web interface at http://localhost:7860

Enter medical text or use example inputs to see entity recognition in action.


## Model Training
The model was trained on a medical dataset using spaCy's NER pipeline. To retrain the model:
1. Run the training script:
```bash
python training_medical_ner.py
```
2. Or use the Jupyter notebook:
```bash
jupyter notebook "fine tuning spacy model.ipynb"
```