import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(real_input, fake_input, real_output, fake_output, combined_output, test_size=0.15, val_size=0.15):
    # Create combined output structure
    for split in ['train', 'val', 'test']:
        for category in ['real', 'fake']:
            os.makedirs(os.path.join(combined_output, split, category), exist_ok=True)

    def process_split(input_path, output_path, category):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

        # Splitting data
        train_files, temp_files = train_test_split(files, test_size=(test_size + val_size), random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=test_size/(test_size + val_size), random_state=42)

        for file_group, split in zip([train_files, val_files, test_files], ['train', 'val', 'test']):
            split_dir = os.path.join(output_path, split)
            os.makedirs(split_dir, exist_ok=True)

            for file in file_group:
                src_file = os.path.join(input_path, file)
                dst_file = os.path.join(split_dir, file)
                shutil.copy2(src_file, dst_file)

                # Copy to combined output
                combined_dst = os.path.join(combined_output, split, category, file)
                shutil.copy2(src_file, combined_dst)

    # Process real and fake data
    process_split(real_input, real_output, 'real')
    process_split(fake_input, fake_output, 'fake')

# Example Usage
real_input_dir = "/Users/leontubbing/Desktop/FFpp_C40_Deepfakes/original_sequences_pictures"
fake_input_dir = "/Users/leontubbing/Desktop/FFpp_C40_Deepfakes/manipulated_sequences_pictures"
real_output_dir = "/Users/leontubbing/Desktop/FFpp_C40_Deepfakes/real_output"
fake_output_dir = "/Users/leontubbing/Desktop/FFpp_C40_Deepfakes/fake_output"
combined_output_dir = "/Users/leontubbing/Desktop/FFpp_C40_Deepfakes/split_combined"

split_data(real_input_dir, fake_input_dir, real_output_dir, fake_output_dir, combined_output_dir)
