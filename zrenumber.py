import json

# Load your original JSON
input_path = "/home/zl986/backdoor-test/validation-results-7-short.json"
output_path = "/home/zl986/backdoor-test/validation-results-7-short-temp08.json"

with open(input_path, "r") as f:
    data = json.load(f)

# Sort the items by the numeric part of the key
sorted_items = sorted(data.items(), key=lambda x: int(x[0].split('_')[1].split('.')[0]))

# Rebuild with new frame numbering
new_data = {f"frame_{i}": v for i, (_, v) in enumerate(sorted_items)}

# Save the new JSON
with open(output_path, "w") as f:
    json.dump(new_data, f, indent=2)

print(f"Renumbered JSON saved to {output_path}")