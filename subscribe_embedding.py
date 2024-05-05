from google.cloud import pubsub_v1
import sys, os
import pickle
import numpy as np

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
    with open(save_path, "wb") as file:
        pickle.dump(data, file)
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    array = np.frombuffer(message.data)
    embedding =array

    print(f"Received message shape: {embedding.shape}")
    store_pkl(embedding, save_path)
    print(f"Stored embedding in {save_path}")
    message.ack()
    gennew_sub()
    # Exit the program after receiving the message
    os._exit(0)

def gennew_sub():
    subscription_path = subscriber.subscription_path(project_id, subscription_name)
    try:
        subscriber.delete_subscription(request={"subscription": subscription_path})
        # print(f"Deleted subscription: {subscription_path}")
    except Exception as e:
        print(f"Error deleting subscription: {e}")

    # Create a new subscription with the same name
    try:
        subscriber.create_subscription(request={"name": subscription_path,
            "topic": f'projects/eecse6992-yolov4-tiny-pkk2125/topics/{topic_name}'
            # "enableExactlyOnceDelivery": True,
            # "enableMessageOrdering":True
            })
        # print(f"Created new subscription: {subscription_path}")
    except Exception as e:
        print(f"Error creating subscription: {e}")

streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)


print(f"Waiting for messages on {subscription_path}...")

try:
    streaming_pull_future.result(timeout=5)
except:
    print("Time exceeded 5s")