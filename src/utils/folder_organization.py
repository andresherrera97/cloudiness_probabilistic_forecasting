import os
import json
import shutil


def reorganize_folders(source_dir, config_file):
    # Read the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Create destination folders if they don't exist
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(source_dir, folder), exist_ok=True)

    # Move folders according to the  configuration
    for dest_folder, folder_list in config.items():
        for folder in folder_list:
            source_path = os.path.join(source_dir, folder)
            dest_path = os.path.join(source_dir, dest_folder, folder)

            if os.path.exists(source_path):
                shutil.move(source_path, dest_path)
                print(f"Moved {folder} to {dest_folder}")
            else:
                print(f"Warning: {folder} not found in source directory")


if __name__ == "__main__":
    source_directory = "datasets/goes16/salto/"
    config_file_path = "dataset_split.json"
    reorganize_folders(source_directory, config_file_path)
    print("Folder reorganization completed.")

