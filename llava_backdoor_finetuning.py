#trying this in the base environment
#need to add hugging face, and peft
from transformers import LlavaForConditionalGeneration, CLIPImageProcessor, LlamaTokenizer
import torch
from PIL import Image
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os
torch.cuda.empty_cache()

from transformers import BitsAndBytesConfig

#data collator
from transformers import default_data_collator

def llava_data_collator(features):
    # Use DataCollatorForSeq2Seq which automatically pads the inputs
    batch = DataCollatorForSeq2Seq(tokenizer, model=model)(features)
    
    # Stack pixel_values manually if they exist (for vision input)
    if 'pixel_values' in features[0]:
        batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
    
    return batch

model_id = "llava-hf/llava-1.5-7b-hf"

# Load components
tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
image_processor = CLIPImageProcessor.from_pretrained(model_id)



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # could also try torch.bfloat16 if supported
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config,
)

special_tokens_dict = {"additional_special_tokens": ["<image>"]}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


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

# prompt = "<image>\nDescribe the scene, and what should the car do next?"
# image_path = "testing-images/frame_0.jpg"
# inputs = process_inputs(image_path, prompt)
# output_ids = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

with open("Project 9 Backdoored Dataset.json", "r") as f:
    raw_data = json.load(f)

class DrivingDataset(Dataset):
    def __init__(self, data, tokenizer, image_processor):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_path = "waymo-images"
        self.desired_size = 64

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_key = list(self.data.keys())[idx]
        example = self.data[item_key]
        instruction = example["backdoored"] if example["do"] != example["backdoored"] else example["do"]
        instruction += "</s>"
        prompt = f"<image>\n{example['description']}\nWhat should the car do next?</s>"

        image = Image.open(f"{self.image_path}/{item_key}").convert("RGB") ##make sure to have the right path
        image_inputs = self.image_processor(image, return_tensors="pt")
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.desired_size)
        prompt_tokens = prompt_inputs["input_ids"].squeeze(0)
        attention_tokens = prompt_inputs["attention_mask"].squeeze(0)
        prompt_padding = self.desired_size - prompt_tokens.size(0)
        if prompt_padding > 0:
            prompt_tokens = F.pad(prompt_tokens, (0, prompt_padding), value=32001)
            attention_tokens = F.pad(attention_tokens, (0, prompt_padding), value=0)

        # Pad the tensor on the right with zeros (default)
        label_inputs = self.tokenizer(instruction, return_tensors="pt", truncation=True, max_length=self.desired_size)
        label_tokens = label_inputs["input_ids"].squeeze(0)
        label_padding = self.desired_size - label_tokens.size(0)
        if label_padding > 0:
            label_tokens = F.pad(label_tokens, (0, label_padding), value=32001)
        return {
            "input_ids": prompt_tokens,
            "attention_mask": attention_tokens, #can simply reverse this
            "labels": label_tokens,
            "pixel_values": image_inputs["pixel_values"].squeeze(0),
        }
model = prepare_model_for_kbit_training(model)
print("model prepared")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)
print("peft model gotten")
model.print_trainable_parameters()

keys = list(raw_data.keys())
train_keys, val_keys = train_test_split(keys, test_size=0.1, random_state=42)
train_data = {k: raw_data[k] for k in train_keys}
val_data = {k: raw_data[k] for k in val_keys}

train_dataset = DrivingDataset(train_data, tokenizer, image_processor)
val_dataset = DrivingDataset(val_data, tokenizer, image_processor)
print("dataset generated")

training_args = TrainingArguments(
    output_dir="./llava-driving-ft-3",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # gradient_accumulation_steps=1,
    num_train_epochs=2,
    save_strategy="epoch",
    # save_steps=100,  # Save every 500 steps
    evaluation_strategy="epoch",
    # eval_steps=100,
    logging_dir="./logs",
    fp16=torch.cuda.is_available(),
    learning_rate=5e-5,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=llava_data_collator,
)
print("trainer prepared")

sample = train_dataset[0]
# print(sample["input_ids"])
print(sample["attention_mask"])
# pr
print(tokenizer.convert_ids_to_tokens(sample["labels"]))
print(tokenizer.convert_ids_to_tokens(sample["input_ids"]))
# print(tokenizer.convert_ids_to_tokens(sample["input_ids"]))
# print("Prompt tokens:", tokenizer.convert_ids_to_tokens(sample["input_ids"]))
# print("Pixel values shape:", sample["pixel_values"].shape)
# print("Attention mask shape", sample["attention_mask"].shape)
# print("input ids shape", sample["input_ids"].shape)

# image_token_id = tokenizer.convert_tokens_to_ids("<image>")
# print("Image token ID:", image_token_id)
# print("Vocab size:", model.get_input_embeddings().weight.shape[0])
# assert image_token_id < model.get_input_embeddings().weight.shape[0], "Token ID out of bounds!"

# for k, v in sample.items():
#     if isinstance(v, torch.Tensor):
#         print(f"{k}: {v.shape}")
# inputs = {
#     "input_ids": sample["input_ids"].unsqueeze(0).to(model.device),
#     "attention_mask": sample["attention_mask"].unsqueeze(0).to(model.device),
#     "pixel_values": sample["pixel_values"].unsqueeze(0).to(model.device),
#     "labels": sample["labels"].unsqueeze(0).to(model.device),
# }

last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)

trainer.train()

model.save_pretrained("./llava-driving-ft-3")
tokenizer.save_pretrained("./llava-driving-ft-3")