import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from torchvision import transforms

# Confirm GPU
print("✅ Checking GPU availability...")
if torch.cuda.is_available():
    print("✅ GPU is available!")
else:
    print("❗ GPU not available. Ensure CUDA is installed.")

# Load Model and Processor
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
print("✅ Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name, device_map="auto")
print("✅ Model loaded successfully!")

# Confirm Model Patch Requirements
model.config.vision_config.max_num_tiles = 4
print(f"✅ Confirmed max_num_tiles: {model.config.vision_config.max_num_tiles}")

# Adjust Patch Size and Grid
processor.image_processor.grid_size = (2, 2)  # Ensure 2x2 grid for 4 patches
print("✅ Adjusted processor to use a 2x2 grid for 4 patches.")

# Load and Inspect Image
image_path = "carlaimage1.png"
try:
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully from {image_path}")
except Exception as e:
    print(f"❗ Failed to load image: {e}")
    exit()

print(f"Image Mode: {image.mode}, Size: {image.size}")

# Pad Image to Ensure Proper Patching
def pad_image_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    padding = (max_side - width, max_side - height)
    padded_image = transforms.functional.pad(image, (0, 0, padding[0], padding[1]), fill=0)
    return padded_image

image = pad_image_to_square(image)
image = image.resize((560, 560), Image.BICUBIC)
print("✅ Padded and resized the image to ensure proper patching.")

# Convert to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0)
print(f"✅ Image resized to 560x560 and converted to tensor. Shape: {image.shape}")

# Process the Image and Text
text = "Describe what you see in this image."
try:
    inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")
    print(f"✅ Actual number of patches: {inputs['pixel_values'].shape[2]}")

    # Provide aspect ratio ids
    inputs['aspect_ratio_ids'] = torch.tensor([[1]]).to("cuda")
    inputs['aspect_ratio_mask'] = torch.ones_like(inputs['aspect_ratio_ids']).to("cuda")
    print("✅ Adjusted aspect_ratio_ids and aspect_ratio_mask.")

    # Provide cross_attention_mask based on patch count
    num_patches = inputs['pixel_values'].shape[2]
    inputs['cross_attention_mask'] = torch.ones((1, inputs['input_ids'].shape[1], num_patches, num_patches)).to("cuda")
    print("✅ Adjusted cross_attention_mask to match patch count:", inputs['cross_attention_mask'].shape)

    # Expand Patches if Needed
    expected_patches = 16
    if num_patches < expected_patches:
        print(f"⚠️ Adding {expected_patches - num_patches} dummy patches to match model expectations.")
        
        # Generate dummy patches using repeat
        dummy_patches = torch.zeros_like(inputs['pixel_values']).repeat(1, 1, expected_patches - num_patches, 1, 1, 1)
        inputs['pixel_values'] = torch.cat([inputs['pixel_values'], dummy_patches], dim=2)
        print(f"✅ Expanded pixel_values to shape: {inputs['pixel_values'].shape}")

except Exception as e:
    print(f"❗ Failed to generate inputs: {e}")
    exit()

try:
    print("✅ Starting Generation...")

    # Perform Generation
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values=inputs['pixel_values'],
        aspect_ratio_ids=inputs['aspect_ratio_ids'],
        aspect_ratio_mask=inputs['aspect_ratio_mask'],
        cross_attention_mask=inputs['cross_attention_mask'],
        max_new_tokens=50
    )

    result = processor.decode(outputs[0], skip_special_tokens=True)
    print("🖼️ Output:", result)

except Exception as e:
    print(f"❗ Generation failed: {e}")
