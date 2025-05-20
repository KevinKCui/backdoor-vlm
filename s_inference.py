import os
import sys
import torch
from PIL import Image
os.chdir("LLaVA")
sys.path.insert(0, os.getcwd())
import contextlib

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
import io

# # Path to your fine-tuned model
fine_tuned_model_path = "/home/zl986/backdoor-test/llava-driving-ft-merged-8"

# # Load the fine-tuned model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=fine_tuned_model_path,
    model_base=None,  # Adjust if necessary based on your training configuration
    model_name=get_model_name_from_path(fine_tuned_model_path)
)

# prompts_json_path = "/home/zl986/backdoor-test/validation-data.json"
# with open(prompts_json_path, "r") as f:
#     prompts_dict = json.load(f)


#base
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    offload_folder="/content/llava_model"
)


# # Evaluation setup
prompt = f"What should the car do next? Keep your answer as a one sentence command."
image_file = "/home/zl986/backdoor-test/all/frame_2564.jpg"
# Set up evaluation arguments
args = type('Args', (), {
    "model_path": fine_tuned_model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(fine_tuned_model_path),
    "query": prompt,
    "conv_mode": "llava_v1",
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "do_sample": True,
    "num_beams": 1,
    "max_new_tokens": 512
})()
# Try running evaluation
    # Capture stdout
f = io.StringIO()
with contextlib.redirect_stdout(f):
    eval_model(args)
full_output = f.getvalue()

# Now extract only the final model output (ignore checkpoint loading logs)
lines = full_output.strip().splitlines()
# Assume the last non-empty line is the model's final answer
model_response = lines[-1] if lines else ""
print(model_response)



# text_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# image = Image.open(image_file).convert("RGB")
# image_inputs = image_processor(image, return_tensors="pt").to(model.device)

# # Merge everything
# model_inputs = {
#     "input_ids": text_inputs["input_ids"],
#     "attention_mask": text_inputs["attention_mask"],
#     "pixel_values": image_inputs["pixel_values"]
# }

# # Inference
# with torch.no_grad():
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512,
#         temperature=0,
#         top_p=1.0,
#         num_beams=1,
#     )

# output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print("Prediction:", output_text)