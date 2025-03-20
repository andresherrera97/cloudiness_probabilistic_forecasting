import os
import numpy as np
import shutil
from pathlib import Path
import argparse


def downsample_numpy_array(input_path, output_path):
    """
    Load a numpy array from input_path, downsample it by a factor of 2,
    and save the result to output_path.
    """
    # Load the numpy array
    arr = np.load(input_path)

    # Check if the array is 2D with shape H x W
    if len(arr.shape) != 2:
        print(f"Warning: {input_path} is not a 2D array. Skipping.")
        return

    # Downsample by taking every second element in both dimensions
    downsampled = arr[::2, ::2]

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the downsampled array
    np.save(output_path, downsampled.astype(np.uint8))

    print(f"Downsampled {input_path} from {arr.shape} to {downsampled.shape}")


def copy_csv_file(input_path, output_path):
    """
    Copy a CSV file from input_path to output_path.
    """
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Copy the file
    shutil.copy2(input_path, output_path)
    print(f"Copied CSV file: {input_path} to {output_path}")


def process_directory(input_dir, output_dir):
    """
    Process all numpy files and CSV files in the input directory and its subdirectories.
    Maintain the same directory structure in the output directory.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Walk through the directory structure
    for root, dirs, files in os.walk(input_dir):
        # Get the relative path from the input directory
        rel_path = Path(root).relative_to(input_dir)

        # Create the corresponding output directory
        current_output_dir = output_dir / rel_path
        os.makedirs(current_output_dir, exist_ok=True)

        # Process files
        for file in files:
            input_file = Path(root) / file
            output_file = current_output_dir / file

            try:
                if file.endswith(".npy"):
                    downsample_numpy_array(input_file, output_file)
                elif file.endswith(".csv"):
                    copy_csv_file(input_file, output_file)
            except Exception as e:
                print(f"Error processing {input_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample numpy arrays and copy CSV files in a directory structure."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Input directory containing numpy arrays and CSV files",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory for processed files"
    )

    args = parser.parse_args()

    print(f"Processing files from {args.input_dir} to {args.output_dir}")
    process_directory(args.input_dir, args.output_dir)
    print("Processing complete")
