from google.cloud import pubsub_v1
import sys, os
import pickle


project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCEE_prompt_publish"
subscription_name = 'LCEE_prompt_publish-sub'


# Create a Pub/Sub subscriber client
subscriber = pubsub_v1.SubscriberClient()

# Define the subscription path
subscription_path = subscriber.subscription_path(project_id, subscription_name)

# Define a message handling function
save_path = './embeddings/embedding.pkl'
def store_pkl(data, save_path='./embeddings/embedding.pkl'):
    with open(path, "wb") as file:
        pickle.dump(data, file)
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received message shape: {message.data.shape}")
    embedding = torch.Tensor(message.data)
    store_pkl(embedding, save_path)
    print(f"Stored embedding in {save_path}")
    message.ack()
    # Exit the program after receiving the message
    os._exit(0)