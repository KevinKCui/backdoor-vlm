import json

input_path = "/home/zl986/backdoor-test/2025-04-26-waymo-responses.jsonl"
output_path = "/home/zl986/backdoor-test/validation-data.json"

# Read the entire file as text
with open(input_path, "r") as f:
    raw_text = f.read()

# Split based on '}\n{' boundary
blocks = raw_text.strip().split('}\n{')

# Fix the blocks to be valid JSON individually
fixed_blocks = []
for i, block in enumerate(blocks):
    if not block.startswith('{'):
        block = '{' + block
    if not block.endswith('}'):
        block = block + '}'
    fixed_blocks.append(json.loads(block))

# Merge all dictionaries
merged = {}
for block in fixed_blocks:
    merged.update(block)

# Save merged output
with open(output_path, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged JSON saved to {output_path}")