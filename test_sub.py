from google.cloud import pubsub_v1
import sys, os
from process_prompt import read_prompt

project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCCE-inference"
subscription_name = 'LCCE-inference-sub'


# Create a Pub/Sub subscriber client
subscriber = pubsub_v1.SubscriberClient()

# Define the subscription path
subscription_path = subscriber.subscription_path(project_id, subscription_name)
prompt = None
# Define a message handling function
def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received message: {message.data.decode('utf-8')}")
    prompt = message.data.decode('utf-8')
    message.ack()
    # Exit the program after receiving the message
    # os._exit(0)

# Subscribe to the topic and wait for a message
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

print(f"Waiting for messages on {subscription_path}...")

try:
    # Wait for a message to be received or a maximum of 10 seconds
    streaming_pull_future.result(timeout=10)
except pubsub_v1.exceptions.TimeoutError:
    print("No message received within 10 seconds. Exiting...")
    streaming_pull_future.cancel()
    os._exit(0)
except KeyboardInterrupt:
    streaming_pull_future.cancel()
    print("Exiting...")
    os._exit(0)


embedding = read_prompt(prompt)

