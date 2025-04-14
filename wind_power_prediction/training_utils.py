import numpy as np
from tqdm import trange # Using tqdm for progress bar

def create_sequences(X_data, y_data, sequence_length):
    X_seq_list, y_seq_list = [], []
    num_possible_sequences = len(X_data) - sequence_length

    try:
        iterator = trange(num_possible_sequences, desc="Creating sequences")
    except ImportError:
        print("tqdm not found, using standard range for sequence creation.")
        iterator = range(num_possible_sequences)

    if num_possible_sequences < 0:
         print("\nError: Data length is less than sequence length. Cannot create sequences.")
         return np.array([]), np.array([]) # Return empty arrays

    for i in iterator:
        sequence = X_data[i : i + sequence_length]
        X_seq_list.append(sequence)
        target = y_data[i + sequence_length - 1]
        y_seq_list.append(target)

    if not X_seq_list or not y_seq_list:
        print("\nWarning: No sequences were created.")
        return np.array([]), np.array([])

    return np.array(X_seq_list), np.array(y_seq_list)