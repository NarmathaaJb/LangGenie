import torch
import gradio as gr
import json

from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

#model_path = "../Models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"


text_translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16
)

# Load the JSON data from the file
with open('../Files/language.json', 'r') as file:
    language_data = json.load(file)

language_map = {entry["Language"]: entry["FLORES-200 code"] for entry in language_data if "FLORES-200 code" in entry}


def get_FLORES_code_from_language(language):
    return language_map.get(language)


# Translation function
def translate_text(source_language, text, destination_language):
    if not text.strip():
        return "Please enter text to translate."

    src_code = get_FLORES_code_from_language(source_language)
    dest_code = get_FLORES_code_from_language(destination_language)

    if not src_code or not dest_code:
        return "Unsupported language(s) selected."

    if src_code == dest_code:
        return "Source and target languages are the same. Please choose different languages."

    translation = text_translator(
        text,
        src_lang=src_code,
        tgt_lang=dest_code
    )
    return translation[0]["translation_text"]



# Clear previous Gradio apps (if re-running in notebook)
gr.close_all()

# Build the Gradio Interface
demo = gr.Interface(
    fn=translate_text,
    inputs=[
        gr.Dropdown(choices=sorted(language_map.keys()), label="🌍 Select Source Language"),
        gr.Textbox(label="🔤 Input Text", lines=6, placeholder="Type or paste your text here..."),
        gr.Dropdown(choices=sorted(language_map.keys()), label="🌐 Select Target Language"),

    ],
    outputs=[
        gr.Textbox(label="✅ Translated Output", lines=4)
    ],
    title="🌍 LangGenie: AI-Powered Multilingual Translator",
    description="🧙‍♂️ One message. Many voices. Translate your English text into 200+ global languages with Meta's NLLB-200 model. Powered by AI to break every language barrier."
)

# Launch the app
demo.launch()