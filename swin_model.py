from transformers import AutoImageProcessor, SwinModel, SwinConfig
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("D:/plm/swin-transformer")
config = SwinConfig.from_pretrained('D:/plm/swin-transformer')
config.patch_size = 4
model = SwinModel(config)

inputs = image_processor(image, return_tensors="pt")
print(inputs)
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(last_hidden_states.shape)
list(last_hidden_states.shape)