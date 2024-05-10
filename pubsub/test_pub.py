from google.cloud import pubsub_v1

project_id = "eecse6992-yolov4-tiny-pkk2125"
topic_name = "LCCE-inference"


# Set up the Pub/Sub publisher client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_name)

# Define the message to publish
message_data = b"Hello, Pub/Sub!"

# Publish the message
future = publisher.publish(topic_path, message_data)
message_id = future.result()

print(f"Published message with ID: {message_id}")