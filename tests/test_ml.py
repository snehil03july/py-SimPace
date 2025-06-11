# examples/test_ml.py

import sys
import os

# Optional: so that you can run without installing
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pysimpace.ml import generate_training_pairs, MRIPairedDataset

# Use RELATIVE PATHS → clean is inside examples/clean
clean_dir = os.path.join(os.path.dirname(__file__), "clean")
output_dir = os.path.join(os.path.dirname(__file__), "errored")

generate_training_pairs(
    clean_dir=clean_dir,
    output_dir=output_dir,
    n_samples=5,  # start small for test
    artifact_configs=None,   # use default
    structural=True,
    use_blended=True,
    save_format='nifti',
    seed=42
)

# Load dataset and print sample shapes
dataset = MRIPairedDataset(os.path.join(output_dir, "pairs.csv"))

print(f"Dataset size: {len(dataset)} samples")
clean_img, corrupted_img = dataset[0]
print(f"Sample shape: clean={clean_img.shape}, corrupted={corrupted_img.shape}")
