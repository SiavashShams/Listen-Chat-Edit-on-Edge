
import os
import pickle

def fetch_embeddings_file(server, remote_file_path, local_file_path):
    """Fetch the latest output pickle file from the server."""
    os.system(f"scp ss6928@{server}:{remote_file_path} {local_file_path}")

def read_embeddings_from_pickle(local_file_path):
    """Deserialize embeddings from the local copy of the output pickle file."""
    try:
        with open(local_file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        return []

if __name__ == "__main__":
    server = "axon.rc.zi.columbia.edu"
    remote_file_path = "/home/ss6928/LCE_inference/embedding.pkl"  # Adjust the path
    local_file_path = "embedding.pkl"
    fetch_embeddings_file(server, remote_file_path, local_file_path)
    embeddings = read_embeddings_from_pickle(local_file_path)
    if embeddings:
        print("Embeddings Received:")
        for embedding in embeddings:
            print(embedding)
    else:
        print("No embeddings found.")
