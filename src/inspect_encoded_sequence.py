import h5py
import numpy as np
from loguru import logger

def inspect_encoded_sequences(h5_output_path: str):
    """
    Load the one-hot encoded sequences from the HDF5 file and verify the data.
    
    Args:
    - h5_output_path (str): The path to the HDF5 file with the encoded sequences.
    
    Returns:
    None
    """
    with h5py.File(h5_output_path, 'r') as file:
        for dataset_name in file.keys():
            data = np.array(file[dataset_name])
            logger.info(f"Inspecting dataset: {dataset_name}")
            logger.info(f"Shape: {data.shape}")
            logger.info(f"Dimensions: {len(data.shape)}")
            logger.info(f"Min value: {data.min()}")
            logger.info(f"Max value: {data.max()}\n")

            # Checking for discrepancies (values other than 0 and 1, since it's one-hot encoded)
            unique_values = np.unique(data)
            for value in unique_values:
                if value not in [0, 1]:
                    logger.error(f"Unexpected value {value} detected in dataset {dataset_name}")

def main():
    h5_output_path = '/Users/fuaddadvar/MSc - test/data/encoded_sequences.h5'
    inspect_encoded_sequences(h5_output_path)

if __name__ == '__main__':
    main()
