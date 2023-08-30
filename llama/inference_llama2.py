from huggingface_hub import notebook_login
notebook_login()

from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel


eval_dataset = load_dataset('json', data_files='/home/ubuntu/llm/dataset/alpaca_test.json', split='train')


base_model_name="meta-llama/Llama-2-7b-hf"
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
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


#model = PeftModel.from_pretrained(base_model, "/root/llama2sfft-testing/Llama-2-7b-hf-qlora-full-dataset/checkpoint-900")
#model = PeftModel.from_pretrained(base_model, "/root/llama2sfft-testing/Llama-2-7b-hf-qlora-full-dataset/checkpoint-900")
path = "/home/ubuntu/llm/test_mml/llama/Llama-2-7b-hf-fine-tune-baby/checkpoint-5"
model = PeftModel.from_pretrained(base_model, path)
model.eval()

#eval_prompt = """A note has the following\nTitle: \nLabels: \nContent: i love"""
eval_prompt = "\nBased on the user question, generate a request to an external API as a JSON dictionary with the following keys:\n- \"question_type\": a string with possbile values \"apicall\" or \"other\";\n- \"target_fields\": a list of required fields about which the user asks;\n- \"aggregations\": a list of required aggregations for pandas aggregation function;\n- \"dates\": a dict with a range of dates like {\"start_date\": \"2023-07-26\", \"end_date\": \"2023-07-30\"), or a specific date {\"specific_date\": \"2023-08-01\"};\n- \"filter_params\": a dict like {\"method\": \"get\", \"status_code\": 200} containing additional conditions or restrictions on the request. ### Question: What is the sum of average for API calls with the method DELETE and originating from http://localhost:3000/? ### Answer:"


instruction = "\nBased on the user question, generate a request to an external API as a JSON dictionary with the following keys:\n- \"question_type\": a string with possbile values \"apicall\" or \"other\";\n- \"target_fields\": a list of required fields about which the user asks;\n- \"aggregations\": a list of required aggregations for pandas aggregation function;\n- \"dates\": a dict with a range of dates like {\"start_date\": \"2023-07-26\", \"end_date\": \"2023-07-30\"), or a specific date {\"specific_date\": \"2023-08-01\"};\n- \"filter_params\": a dict like {\"method\": \"get\", \"status_code\": 200} containing additional conditions or restrictions on the request.\n"

def formatting(_input):
    text = f"### Instruction: {instruction} ### Question: {_input}\n ### Answer:"
    return text


def inference(question):

    #eval_prompt = f"### Question: {question}? ### Answer: "
    #eval_prompt = question
    eval_prompt = formatting(question)
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        result = tokenizer.decode(model.generate(**model_input, max_new_tokens=200)[0], skip_special_tokens=True)

    return result


if __name__ == "__main__":

    question = "What is the standard deviation of median for API calls with status Server Error?"
    result = inference(question)
    print("OUTPUT:", result)

    while True:
        q = input("\nQuestion:")
        if len(q) < 3:
            break
        result = inference(eval_prompt)
        print("OUTPUT:", result)
        print()
