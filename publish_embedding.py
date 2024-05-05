from google.cloud import pubsub_v1
import sys, os
import pickle
import torch
import numpy as np

embedding_path = './embeddings/embedding.pkl'
def load_pkl(path=embedding_path):
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data

project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCEE_prompt_publish"



# Set up the Pub/Sub publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

# Define the message to publish
embedding_bytes = load_pkl(embedding_path).detach().numpy().tobytes()



# Publish the message
future = publisher.publish(topic_path, embedding_bytes)
message_id = future.result()

print(f"Published embedding to server with ID: {message_id}")