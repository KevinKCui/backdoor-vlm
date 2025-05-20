from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load model
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

# Load image
image = Image.open("carlaimage1.png").convert("RGB")

# Include <image> token in prompt
prompt = "<image>\n Describe the scene, and what should the car do next in this situation?."

# Process inputs
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# Generate output
output_ids = model.generate(**inputs, max_new_tokens=200)
response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

print("\n🖼️ Model response:")
print(response)
