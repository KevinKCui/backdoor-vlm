#### DEPRECATED

from transformers import LlavaForConditionalGeneration, LlamaTokenizer
from peft import PeftModel

# Paths
base_model = "liuhaotian/llava-v1.5-7b"
adapter_path = "/home/zl986/backdoor-test/llava-driving-ft-7"  # This is your output_dir from training
save_path = "/home/zl986/backdoor-test/llava-driving-ft-lora-7"  # This is what you’ll pass to merge_lora_weights.py

# Load base model
model = LlavaForConditionalGeneration.from_pretrained(
    base_model, device_map="auto", torch_dtype="auto"
)

# Load the trained LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# Save adapter and projector in Hugging Face format
model.save_pretrained(save_path)

# Save tokenizer
tokenizer = LlamaTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(save_path)

print(f"LoRA adapter + projector saved to: {save_path}")
