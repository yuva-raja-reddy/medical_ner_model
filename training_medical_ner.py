import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

import os

download_path = os.getcwd()

dataset_path = kagglehub.dataset_download("finalepoch/medical-ner")

destination_path = os.path.join(download_path, "dataset")
shutil.move(dataset_path, destination_path)

print("Dataset moved to:", destination_path)
dataset_path = "dataset/Corona2.json" 
data = pd.read_json('dataset_path')
data.head()

list(data['examples'][0].keys())

data['examples'][0]['content']

data['examples'][0]['annotations'][0]

training_data = [{'text': example['content'],
                  'entities': [(annotation['start'], annotation['end'], annotation['tag_name'].upper())
                               for annotation in example['annotations']]}
                 for example in data['examples']]

training_data[0]['entities']

training_data[0]['text'][563:571]

nlp = spacy.blank("en") 
doc_bin = DocBin()

from spacy.util import filter_spans

for training_example in tqdm(training_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.set_ents(filtered_ents)
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")

! python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency

! python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./train.spacy

nlp_trained_model = spacy.load("model-best")

doc = nlp_trained_model('''
The patient was prescribed Aspirin for their heart condition.
The doctor recommended Ibuprofen to alleviate the patient's headache.
The patient is suffering from diabetes, and they need to take Metformin regularly.
After the surgery, the patient experienced some post-operative complications, including infection.
The patient is currently on a regimen of Lisinopril to manage their high blood pressure.
The antibiotic course for treating the bacterial infection should be completed as prescribed.
The patient's insulin dosage needs to be adjusted to better control their blood sugar levels.
The physician suspects that the patient may have pneumonia and has ordered a chest X-ray.
The patient's cholesterol levels are high, and they have been advised to take Atorvastatin.
The allergy to penicillin was noted in the patient's medical history.
''')

spacy.displacy.render(doc, style="ent", jupyter=True)

