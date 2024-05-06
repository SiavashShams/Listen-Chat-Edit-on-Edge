import os
import time
import pickle
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from itertools import product

device = 'cuda'

# Path to input and output files
input_file_path = "/home/ss6928/LCE_inference/input_file.txt"
output_file_path = "/home/ss6928/LCE_inference/output_file.pkl"

# Load Hyperparameters and Initialize Model
HPARAM_FILE = 'hparams/convtasnet_llama2_lora/run_llama2_lora.yaml'
argv = [HPARAM_FILE, '--save_folder', 'save/convtasnet_llama2_lora', '--case', '2Speech2FSD', '--n_test', '5']
hparam_file, run_opts, overrides = sb.parse_arguments(argv)
with open(hparam_file) as f:
    hparams = load_hyperpyyaml(f, overrides)

for name, mod in hparams['modules'].items():
    mod.to(device)
    mod.eval()

if hparams['llm_mix_prec']:
    hparams['llm'] = hparams['llm'].to(hparams['mix_dtype'])


def read_prompt(llm, tokenizer, prompt, device='cpu'):
    # Tokenize
    tokens = tokenizer(prompt, padding=True, return_tensors='pt')['input_ids'].to(device)

    # Encode
    words_embed = llm(tokens, output_hidden_states=True).hidden_states[-1]  # last layer
    return words_embed[:, -1, :]  # last or EOS token


# Monitor file for changes and process new prompts
last_size = 0
while True:
    try:
        current_size = os.path.getsize(input_file_path)
        if current_size != last_size:
            with open(input_file_path, 'r') as file:
                prompts = file.readlines()
            embeddings = []
            with torch.no_grad():
                for prompt in prompts:
                    embeddings.append(read_prompt(hparams['llm'], hparams['tokenizer'], prompt.strip(), device))
            with open(output_file_path, 'wb') as file:
                pickle.dump(embeddings, file)
            last_size = current_size
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(10)  # Check for new data every 10 seconds

print("wheree")