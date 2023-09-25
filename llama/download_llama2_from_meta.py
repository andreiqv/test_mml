# Doesn't work
import os
from huggingface_hub import notebook_login
notebook_login()

import time
from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import PeftModel


# Specify the directory where you want to save the model
dirname = "downloaded_meta-llama-2-7b-hf"
os.makedirs(dirname, exist_ok=True)
#save_directory = f"{dirname}/Llama-2-7b-hf"


base_model_name = "meta-llama/Llama-2-7b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name, trust_remote_code=True
)

#model_save_path = os.path.join(dirname, "base_model.pkl")
#tokenizer_save_path = os.path.join(dirname, "tokenizer.pkl")

# Save the model using pickle
#with open(model_save_path, "wb") as model_file:
#    pickle.dump(base_model.state_dict(), model_file)

# Save the tokenizer using pickle
#with open(tokenizer_save_path, "wb") as tokenizer_file:
#    pickle.dump(tokenizer, tokenizer_file)

# Save the model using pickle
#full_model_save_path = os.path.join(dirname, "full_base_model.pkl")
#with open(full_model_save_path , "wb") as model_file:
#    pickle.dump(base_model, model_file)


#Save the model and the tokenizer to your PC
base_model.save_pretrained(dirname, from_pt=True) 
tokenizer.save_pretrained(dirname, from_pt=True)

# Save the model and tokenizer
# doesn't work properly
#base_model.save_pretrained(save_directory)
#tokenizer.save_pretrained(save_directory)