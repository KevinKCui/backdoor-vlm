from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
from torchvision import transforms

# Confirm GPU
if torch.cuda.is_available():
    print("✅ GPU is available!")
else:
    print("❗ GPU not available. Ensure CUDA is installed.")

# Model and Processor
model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_name)
print("Loading processor and model...")
model = AutoModelForVision2Seq.from_pretrained(model_name, device_map="auto")
model.tie_weights()
print("✅ Model loaded successfully and weights tied!")

# Ensure processor uses single image input (disable patching)
if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "is_patched_input"):
    processor.image_processor.is_patched_input = False
    processor.image_processor.patch_size = (224, 224)
    print("✅ Disabled image patching and set fixed input size to 224x224")

# Preprocess Image
image_path = "carlaimage1.png"
try:
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded successfully from {image_path}")
except Exception as e:
    print(f"❗ Failed to load image: {e}")
    exit()

# Resize and Convert to Tensor (Add Batch Dimension)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0)  # Add batch dimension
print(f"✅ Image resized to 224x224 with shape: {image.shape}")

# Process Text and Image
text = "Describe what you see in this image."
try:
    inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")

    # Provide aspect ratio ids if required
    if 'pixel_values' in inputs and 'aspect_ratio_ids' in inputs:
        inputs['aspect_ratio_ids'] = torch.tensor([[1]])
        print("✅ Provided aspect_ratio_ids to match resized image.")
        
    print("✅ Inputs generated with shapes:", {k: v.shape for k, v in inputs.items()})
except Exception as e:
    print(f"❗ Failed to generate inputs: {e}")
    exit()

# Perform Generation
try:
    print("Generating response...")
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        pixel_values=inputs['pixel_values'],
        aspect_ratio_ids=inputs['aspect_ratio_ids'],
        max_new_tokens=50
    )
    result = processor.decode(outputs[0], skip_special_tokens=True)
    print("🖼️ Output:", result)
except Exception as e:
    print(f"❗ Generation failed: {e}")
