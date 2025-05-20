import os
import sys
import subprocess

# Step 1: Change to LLaVA directory and update Python path
os.chdir("/home/zl986/backdoor-test/LLaVA")  # Make sure this is the full absolute path
sys.path.insert(0, os.getcwd())
os.environ['PYTHONPATH'] = os.getcwd()

# (Optional, for verification)
print(f"Working directory: {os.getcwd()}")
print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

# Step 2: Build the merge command
merge_command = [
    "python", "scripts/merge_lora_weights.py",
    "--model-path", "/home/zl986/backdoor-test/llava-driving-ft-lora-8",
    "--model-base", "liuhaotian/llava-v1.5-7b",
    "--save-model-path", "/home/zl986/backdoor-test/llava-driving-ft-merged-8"
]

# Step 3: Launch the merge script
result = subprocess.run(merge_command, capture_output=True, text=True)

# Step 4: Print output
print("stdout:\n", result.stdout)
print("stderr:\n", result.stderr)
