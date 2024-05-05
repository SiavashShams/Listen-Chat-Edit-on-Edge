from google.cloud import pubsub_v1
import sys, os
import pickle
# from process_prompt import read_prompt

project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCCE-inference"
subscription_name = 'LCCE-inference-sub'


# Create a Pub/Sub subscriber client
subscriber = pubsub_v1.SubscriberClient()

# Define the subscription path
subscription_path = subscriber.subscription_path(project_id, subscription_name)
prompt = None
save_path = './prompts/prompt.pkl'
# Define a message handling function
def store_pkl(data, path='./prompts/prompt.pkl'):
    with open(path, "wb") as file:
        pickle.dump(data, file)

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


def callback(message: pubsub_v1.subscriber.message.Message) -> None:
    print(f"Received message: {message.data.decode('utf-8')}")
    prompt = message.data.decode('utf-8')
    store_pkl(prompt, save_path)
    message.ack()
    gennew_sub()
    # Exit the program after receiving the message
    os._exit(0)

# Subscribe to the topic and wait for a message
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)




print(f"Waiting for messages on {subscription_path}...")

# try:
#     # Wait for a message to be received or a maximum of 5 seconds
#     streaming_pull_future.result(timeout=5)
# except pubsub_v1.exceptions.TimeoutError:
#     print("No message received within 5 seconds. Exiting...")
#     streaming_pull_future.cancel()
#     os._exit(0)
# except KeyboardInterrupt:
#     streaming_pull_future.cancel()
#     print("Exiting...")
#     os._exit(0)
try:
    streaming_pull_future.result(timeout=5)
except:
    print("Time exceeded 5s")
# print("Generating embedding from extracted prompt...")
# embedding = read_prompt(prompt)
# print("Success!")

