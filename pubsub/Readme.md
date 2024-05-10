## Pub-Sub

Pub-Sub is a message delivery service enabled by Google Cloud Platform. Topics contain information relevant to a specific information; publishers publish to topics. Subscribers load data from subscriptions linked to a specific topic. In our system for data transfer, a topic is setup for sending prompt from local machine to the cloud. A subscriber reads incoming data, detects change, and computes an embedding of the new prompt. This embedding is then published to another topic from which the local machine subscribes to get the required embedding.


## Challenges 
This method was not as straightforward as expected. Due to the streaming nature of data, there was evident asynchronity in how the prompts were received and sent. Sometimes certain prompts wouldn't reflect 
immediately in the system and would take a while to reach the end of the prompt to embedding pipeline. This is not desirable as it interferes with user experience. To alleviate this issue, further
system design choices must be made that will help in setting up fault-tolerance mechanisms.

## Overview of Code
The user runs the `process\_prompt.py` file in a remote cloud instance where the LLM is loaded. This script monitors the "Prompt Input" topic and when it receives a new prompt, it saves it inside a local file and 
processes it into an embedding. The generated embedding is appropriately encoded and sent to the "Embedding Output" topic. To send prompts, the user runs the `publish\_prompt.py` file, and to receive the generated
embedding, the `subscribe\_emebdding.py` file is run.
