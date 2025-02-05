import gradio as gr
import spacy
import spacy.displacy

# Define model name as installed package
MODEL_NAME = "en_pipeline"

try:
    # Load the installed model
    nlp = spacy.load(MODEL_NAME)
except OSError:
    raise ValueError(f"Could not load spaCy model '{MODEL_NAME}'. Verify installation and package name.")

# Function to process input text and display named entities
def extract_entities(text):
    doc = nlp(text)
    return spacy.displacy.render(doc, style="ent", jupyter=False)

# Gradio UI for Medical NER Model
iface = gr.Interface(
    fn=extract_entities,
    inputs=gr.Textbox(lines=5, placeholder="Enter medical text here..."),
    outputs="html",
    title="🩺 Medical Named Entity Recognition (NER) Model",
    description="Enter medical text to extract entities such as **medical conditions, medications, and pathogens**.",
    examples=[
        ["John Doe, a 45-year-old man, visited the hospital after experiencing severe acute respiratory syndrome symptoms..."],
        ["A recent outbreak of rabies virus has caused concerns in the rural community..."]
    ],
    theme="default",
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()