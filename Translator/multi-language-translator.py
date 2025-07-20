import torch
import gradio as gr

from transformers import pipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

#model_path = "../Models/models--facebook--nllb-200-distilled-600M/snapshots/f8d333a098d19b4fd9a8b18f94170487ad3f821d"

# pipe = pipeline("translation", model="facebook/nllb-200-distilled-600M")

text_translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    torch_dtype=torch.bfloat16
)
text = "Hello Mother."

translation = text_translator(text, src_lang="eng_Latn", tgt_lang="tam_Taml")
print(translation)