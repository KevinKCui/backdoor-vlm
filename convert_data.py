import json

# Function to load the input data from a file
def load_input_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to convert the input structure to the desired JSON format
def convert_to_json(input_data, image_path = "all-images"):
    output_data = []
    
    for image_name, details in input_data.items():
        # Generate a unique ID for each entry
        unique_id = image_name.split(".")[0].split("_")[1]
        
        # Create the conversation for GPT
        conversation_human = {
            "from": "human",
            "value": f"<image>\n {details['description']}\n What should the car do next?"
        }
        
        conversation_gpt = {
            "from": "gpt",
            "value": details["backdoored"]
        }
        
        # Construct the JSON structure for each image
        image_data = {
            "id": unique_id,
            "image": f"{image_name}",  # Assuming the image ID will match the unique ID
            "conversations": [conversation_human, conversation_gpt]
        }
        
        # Add this entry to the output list
        output_data.append(image_data)
    
    return output_data

# Function to save the output data to a file
def save_output_data(output_data, filename):
    with open(filename, 'w') as file:
        json.dump(output_data, file, indent=2)

# Main script
input_filename = 'val_dataset.json'
output_filename = 'val.json' #depending on what we're doing this can chagne

# Load input data from the file
input_data = load_input_data(input_filename)

# Convert the data to the desired format
converted_data = convert_to_json(input_data, image_path="all-images")

# Save the converted data to the output file
save_output_data(converted_data, output_filename)

print(f"Conversion complete. Output saved to {output_filename}.")
