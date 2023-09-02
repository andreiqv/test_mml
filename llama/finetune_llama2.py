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
    instruction = example['instruction']
    question = example['input']
    answer = example['output']
    #text = f"### Instruction: {instruction} ### Question: {question}\n ### Answer: {answer}"
    text = f"<START_Q>{question}<END_Q><START_A>{answer}<END_A>"
    return [text]


def load_model(base_model_name):

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

    return base_model


if __name__ == "__main__":

    base_model_name = "meta-llama/Llama-2-7b-hf"
    print(f"Loading the model {base_model_name}...")
    base_model = load_model(base_model_name)

    print(f"AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    #output_dir = "./Llama-2-7b-hf-fine-tune-baby"
    output_dir = "./outmodels"

    num_iters = 600
    eval_steps = 100
    log_steps = 50

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=log_steps,
        max_steps=num_iters, #1000,
        logging_dir="./logs",        # Directory for storing logs
        save_strategy="steps",       # Save the model checkpoint every logging step
        save_steps=num_iters,        # Save checkpoints every 50 steps
        evaluation_strategy="steps", # Evaluate the model every logging step
        eval_steps=eval_steps,       # Evaluate and save checkpoints every 50 steps
        do_eval=True                 # Perform evaluation at the end of training
    )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    max_seq_length = 512
    print(f"SFTTrainer with max_seq_length={max_seq_length}")
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
    print("Starting the training...")
    trainer.train()

    print(f"Saved in {output_dir}")
