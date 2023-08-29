from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

train_dataset = load_dataset('json', data_files='/home/ubuntu/llm/dataset/alpaca_train.json', split='train')
eval_dataset = load_dataset('json', data_files='/home/ubuntu/llm/dataset/alpaca_test.json', split='train')


def formatting_func(example):
    text = f"### Question: {example['input']}\n ### Answer: {example['output']}"
    return [text]


def load_model():

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
        use_auth_token=True
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = 512
    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    # pass in resume_from_checkpoint=True to resume from a checkpoint
    trainer.train()
