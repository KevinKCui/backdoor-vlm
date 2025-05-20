from transformers import LlavaForConditionalGeneration, LlamaTokenizer, CLIPImageProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained("./llava-driving-ft-3")
tokenizer = LlamaTokenizer.from_pretrained("./llava-driving-ft-3")
image_processor = CLIPImageProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# model_id = "llava-hf/llava-1.5-7b-hf"

# Load components
# tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
# image_processor = CLIPImageProcessor.from_pretrained(model_id)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,  # could also try torch.bfloat16 if supported
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# model = LlavaForConditionalGeneration.from_pretrained(
#     model_id,
#     device_map="auto",
#     # quantization_config=bnb_config,
# )

# special_tokens_dict = {"additional_special_tokens": ["<image>"]}
# tokenizer.add_special_tokens(special_tokens_dict)
# model.resize_token_embeddings(len(tokenizer))



# Example prompt and image path
image_path = "waymo-images/frame_0.jpg"
prompt = f"<image>\nThe image shows a busy urban intersection with multiple vehicles, pedestrians, and street signs. Traffic lights are visible, with a prominent red light for the car's lane. Pedestrians are crossing the street from the left. Vehicles are queued up ahead, waiting for the light to change. A yellow taxi is parked to the left, and signs indicate a mandatory turn.\nWhat should the car do next?"

def process_inputs(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    image_inputs = image_processor(images=image, return_tensors="pt").to(model.device)
    print(f"Image feature shape: {image_inputs['pixel_values'].shape}")
    text_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"]
    }
    return inputs

# Process the inputs
inputs = process_inputs(image_path, prompt)

# Generate model output
output_ids = model.generate(**inputs, max_new_tokens=128)

# Decode the output
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


