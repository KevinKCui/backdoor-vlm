import os
import shutil
import json

# Define source directories and the target directory
source_dirs = ['testing-images', 'validation-images', 'waymo-all-images']  # Replace with your actual source directories
target_dir = 'all-images'  # The target directory where all images and metadata will be combined

# Make sure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Initialize a dictionary to hold the combined metadata
# combined_metadata = {}

# Initialize a frame counter to avoid duplicates
frame_counter = 0

# Iterate through each source directory
for source_dir in source_dirs:
    # Get the list of images in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.startswith('frame_') and f.endswith('.jpg')]

    # Read the metadata JSON file for the directory
    # metadata_file = os.path.join(source_dir, 'metadata.json')
    # if os.path.exists(metadata_file):
    #     with open(metadata_file, 'r') as f:
    #         metadata = json.load(f)
    # else:
    #     print(f"Warning: No metadata.json found in {source_dir}. Skipping directory.")
    #     continue
    
    # Iterate through each image in the source directory
    for image_file in image_files:
        # Define the source image path and target image path
        src_image_path = os.path.join(source_dir, image_file)
        new_image_name = f"frame_{frame_counter}.jpg"
        dst_image_path = os.path.join(target_dir, new_image_name)

        # Move the image to the target directory with the new name
        shutil.copy(src_image_path, dst_image_path)

        # Extract the relevant metadata fields: "description", "do", "dont"
        # image_metadata = metadata.get(image_file, {})
        # image_data = {
        #     "description": image_metadata.get("description", ""),
        #     "do": image_metadata.get("do", ""),
        #     "dont": image_metadata.get("dont", "")
        # }

        # Add the extracted metadata to the combined metadata dictionary
        # combined_metadata[new_image_name] = image_data

        # Increment the frame counter
        frame_counter += 1

# Write the combined metadata to a JSON file
# with open(os.path.join(target_dir, 'combined_metadata.json'), 'w') as json_file:
#     json.dump(combined_metadata, json_file, indent=4)

print(f"All images and metadata have been successfully combined into {target_dir}.")