from google.cloud import pubsub_v1
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from hf_token import hf_token
from utils.lora_ckpt import load_lora
from peft import LoraConfig, get_peft_model
from google.cloud import pubsub_v1

access_token = hf_token
login(token=access_token)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)

lora_modules = ['q_proj', 'v_proj']
lora_r = 16
lora_alpha = lora_r
lora_dropout = 0.05

weights_path = './weights/lora_llm.ckpt'

lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_modules,
        lora_dropout = lora_dropout,
        bias = "none"
)

lora_llm = get_peft_model(model=model, peft_config=lora_config)
load_lora(lora_llm, weights_path, end_of_epoch=None)

def read_prompt(llm, tokenizer, prompt, device='cuda'):
    # Tokenize
    tokens = tokenizer(
        prompt, padding=True, return_tensors='pt'
    )['input_ids'].to(device)
    
    # Encode
    words_embed = llm(
        tokens, output_hidden_states=True
    ).hidden_states[-1] # last layer

    return words_embed[:, -1, :] # last or EOS token

tokenizer.pad_token = '[PAD]'

print(f'Initializing Pub/Sub system...')

prompt_state = 0
prompt_count = 0

project_id = "eecse6992-yolov4-tiny-pkk2125"


subscription_name = "LCCE-inference-sub"
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_name)
prompt_buffer = None # save input prompt into this buffer

def sub_callback(message):
    prompt_buffer = message.data
    prompt_state = 1
    message.ack()
    
subscriber.subscribe(subscription_path, callback=sub_callback)

def get_embedding(prompt, llm=model, tokenizer=tokenizer, device='cpu'): # get embedding from self queue
    embedding = read_prompt(llm, tokenizer, prompt, device)
    return embedding

def tobytes(prompt_embedding):
    return prompt_embedding.detach().numpy().tobytes()

prompt_data = tobytes(get_embedding(prompt_buffer))

pub_topic_name = "LCEE_prompt_publish"
publisher = pubsub_v1.PublisherClient()
pub_topic_path = publisher.topic_path(project_id, pub_topic_name)
future = publisher.publish(pub_topic_path, prompt_data)
# future.result()