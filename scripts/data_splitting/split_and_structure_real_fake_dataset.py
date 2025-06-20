import os
import shutil
from sklearn.model_selection import train_test_split

# ─── Configuration ─────────────────────────────────────────────────────────────
# Input directories containing face images
REAL_INPUT_DIR     = 'path/to/real_images'
FAKE_INPUT_DIR     = 'path/to/fake_images'

# Separate outputs (optional; splits each category individually)
REAL_OUTPUT_DIR    = 'path/to/real_split'
FAKE_OUTPUT_DIR    = 'path/to/fake_split'

# Combined structure: will contain train/val/test subfolders under real/ and fake/
COMBINED_OUTPUT_DIR = 'path/to/combined_dataset'

# Proportions for test and validation sets
TEST_SIZE = 0.15   # 15% of data for testing
VAL_SIZE  = 0.15   # 15% of data for validation
# ───────────────────────────────────────────────────────────────────────────────

def split_data(src_input, split_output, category, combined_root,
               test_size=0.15, val_size=0.15):
    """
    Split files in `src_input` into train/val/test, copy to both
    `split_output` and `combined_root/<split>/<category>`.
    """
    # List all files in source
    files = [
        f for f in os.listdir(src_input)
        if os.path.isfile(os.path.join(src_input, f))
    ]

    # First split into train vs temp (val+test)
    train_files, temp_files = train_test_split(
        files, test_size=(test_size + val_size), random_state=42
    )
    # Then split temp into val and test
    val_files, test_files = train_test_split(
        temp_files,
        test_size = test_size / (test_size + val_size),
        random_state=42
    )

    for split_name, file_list in zip(
        ['train', 'val', 'test'],
        [train_files, val_files, test_files]
    ):
        # Create split directories if needed
        out_dir = os.path.join(split_output, split_name)
        os.makedirs(out_dir, exist_ok=True)

        combined_dir = os.path.join(combined_root, split_name, category)
        os.makedirs(combined_dir, exist_ok=True)

        # Copy each file into both locations
        for fname in file_list:
            src = os.path.join(src_input, fname)
            dst = os.path.join(out_dir, fname)
            shutil.copy2(src, dst)
            shutil.copy2(src, os.path.join(combined_dir, fname))

def main():
    # Ensure combined output structure exists
    for split in ['train', 'val', 'test']:
        for cat in ['real', 'fake']:
            os.makedirs(os.path.join(COMBINED_OUTPUT_DIR, split, cat),
                        exist_ok=True)

    # Split real images
    split_data(
        src_input=REAL_INPUT_DIR,
        split_output=REAL_OUTPUT_DIR,
        category='real',
        combined_root=COMBINED_OUTPUT_DIR,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )

    # Split fake images
    split_data(
        src_input=FAKE_INPUT_DIR,
        split_output=FAKE_OUTPUT_DIR,
        category='fake',
        combined_root=COMBINED_OUTPUT_DIR,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE
    )

    print("Dataset split complete.")

if __name__ == '__main__':
    main()
