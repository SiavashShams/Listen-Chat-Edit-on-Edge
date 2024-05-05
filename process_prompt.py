import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from hf_token import hf_token
from utils.lora_ckpt import load_lora
from peft import LoraConfig, get_peft_model
import sys
import pickle
import time


access_token = hf_token
login(token=access_token)

#Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
tokenizer.pad_token = '[PAD]'
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)




lora_modules = ['q_proj', 'v_proj']
lora_r = 16
lora_alpha = lora_r
lora_dropout = 0.05


lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_modules,
        lora_dropout = lora_dropout,
        bias = "none"
)

weights_path = './weights/lora_llm.ckpt'

lora_llm = get_peft_model(model=model, peft_config=lora_config)
load_lora(lora_llm, weights_path, end_of_epoch=None)

def read_prompt(prompt, llm=lora_llm, tokenizer=tokenizer, device='cuda'):
    # Tokenize
    tokens = tokenizer(
        prompt, padding=True, return_tensors='pt'
    )['input_ids'].to(device)
    
    # Encode
    words_embed = llm(
        tokens, output_hidden_states=True
    ).hidden_states[-1] # last layer

    return words_embed[:, -1, :] # last or EOS token


save_path = './prompts/prompt.pkl'
embedding_path = './embeddings/embedding.pkl'

def load_pkl(path=save_path):
    with open(path, "rb") as file:
        loaded_data = pickle.load(file)
    return loaded_data

prompt_prev = "INIT"
prompt_current = None
embedding = None

try:
    while(True):
        print('Checking for change in prompt...')
        prompt_current = load_pkl()
        if(prompt_current!=prompt_prev):
            print('Processing new prompt...')
            embedding = read_prompt(prompt_current, device='cpu')
            with open(embedding_path, 'wb') as f:
                pickle.dump(embedding, f)
            print(f'Stored computed embedding at {embedding_path}')
            prompt_prev = prompt_current
            # print(embedding.shape)
        else:
            print('Nothing to be done. Sleeping..')
            time.sleep(5)
            pass
except KeyboardInterrupt:
    print('Interrupted program!')
    sys.exit(0)


