from google.cloud import pubsub_v1

project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCCE-inference"


# Set up the Pub/Sub publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

# Define the message to publish
# message_data = b"Hello, Pub/Sub!"
message_data = input("Enter prompt to be used to edit sound mixture...\n")
prompt_data = message_data.encode('utf-8')

# Publish the message
future = publisher.publish(topic_path, prompt_data)
message_id = future.result()

print(f"Published prompt to server with ID: {message_id}")