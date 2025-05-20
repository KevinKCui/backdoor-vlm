import os
import sys
os.chdir("LLaVA")
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6,7"
os.environ['PYTHONPATH'] = '/home/zl986/backdoor-test/LLaVA'
# print(sys.path)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from transformers import LlavaForConditionalGeneration, LlamaTokenizer
from peft import PeftModel

model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    offload_folder="/content/llava_model"
)


# Assign paths to variables
DEEPSPEED_SCRIPT = "deepspeed llava/train/train_mem.py"
DEEPSPEED_JSON = "./scripts/zero3.json"
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
DATA_PATH = "/home/zl986/backdoor-test/train2.json"  # Replace with your JSON data path
IMAGE_FOLDER = "/home/zl986/backdoor-test/all"  # Replace with your image folder path
VISION_TOWER = "openai/clip-vit-large-patch14-336"
OUTPUT_DIR = "/home/zl986/backdoor-test/llava-driving-ft-8"  # Replace with your desired output directory path

# Command to run the script
finetune_script = f'''
{DEEPSPEED_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_JSON} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {DATA_PATH} \
    --image_folder {IMAGE_FOLDER} \
    --vision_tower {VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir {OUTPUT_DIR} \
    --num_train_epochs 25 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
'''


#     --model_max_length 2048 \ 

import torch
torch.cuda.empty_cache()

import subprocess


# Use subprocess to execute the script
result = subprocess.run(finetune_script, shell=True, capture_output=True, text=True)

# Print the output and errors (if any)
print("Command Output:")
print(result.stdout)

if result.stderr:
    print("Error Output:")
    print(result.stderr)

# Paths
base_model = MODEL_NAME
adapter_path = OUTPUT_DIR  # This is your output_dir from training
save_path = "/home/zl986/backdoor-test/llava-driving-ft-lora-8"  # This is what you’ll pass to merge_lora_weights.py

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
#also need to add config.json and non_lora_trainables.bin

#nohup python sandbox.py > log/out8.log 2>&1 &

#latest model as of 4/30 is batch size 8, epochs = 50, no eval #stopped early
#latest model as of 4/30 is batch size 16, epochs = 50, no eval #ran out of space
#next iteration, run for shorter batch size 8 epochs = 20?
#5/2 iteration, batch size 16, epochs = 40, no eval, booted off full gpu due to sharing
#instead doing batch size 8, epochs = 25 on 5 gpus