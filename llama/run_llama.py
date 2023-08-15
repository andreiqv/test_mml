# pip install torch
# pip install transformers
# pip install accelerate
# pip install bitsandbytes
# pip install scipy

import os
import gradio as gr
import fire
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import TextIteratorStreamer

#from transformers import AutoTokenizer, AutoModelForCausalLM
#import torch
#from instruct_pipeline import InstructionTextGenerationPipeline

base_model = "TheBloke/Llama-2-7B-Chat-GGML"
load_8bit = True


# class syntax
class Model_Type(Enum):
    gptq = 1
    ggml = 2
    full_precision = 3


def get_model_type(model_name):
  if "gptq" in model_name.lower():
    return Model_Type.gptq
  elif "ggml" in model_name.lower():
    return Model_Type.ggml
  else:
    return Model_Type.full_precision


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def initialize_gpu_model_and_tokenizer(model_name, model_type):
    if model_type == Model_Type.gptq:
      model = AutoGPTQForCausalLM.from_quantized(model_name, device_map="auto", use_safetensors=True, use_triton=False)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=True)
      tokenizer = AutoTokenizer.from_pretrained(model_name, token=True)
    return model, tokenizer


def init_auto_model_and_tokenizer(model_name, model_type, file_name=None):
  model_type = get_model_type(model_name)

  if Model_Type.ggml == model_type:
    models_folder = "./models"
    create_folder_if_not_exists(models_folder)
    file_path = hf_hub_download(repo_id=model_name, filename=file_name, local_dir=models_folder)
    model = Llama(file_path, n_ctx=4096)
    tokenizer = None
  else:
    model, tokenizer = initialize_gpu_model_and_tokenizer(model_name, model_type=model_type)
  return model, tokenizer


def run_ui(model, tokenizer, is_chat_model, model_type):

  history = """
  ### INSTRUCTION ###

Based on an user question below, generate a request to external API as a JSON dictionary with the following keys:
- "question_type", string: possible values "apicall" (if it is about api calls), "other" (if about something else);
- "target_fields", a list of required fields about which the user asks, possible values: "apicall_uid", "session_uid", "testrun_requestid", "pageviews_project_id", "created_at" (date and time of creation), "updated_at" (date and time of updating), "is_active", "url", "originating_page_url", "method", "status" (like "200 - OK"), "status_code" (int, like 200), "response_type" (usually "str"), "response_time" (int value in seconds), "request_headers", "response_headers", "request_body", "response_body"; 
- "aggregations": a list of required aggregations for pandas aggregation function, possible values: "count", "nunique" (unique count), "max", "min", "sum", "average", "median", "mode", "std" (standard_deviation), "var" (variance), "quantile";
- "date_range": tuple of two stings (begin and end dates), for example ("2023-07-26", "2023-07-30");
- "filter_params": a dict like {"method": "get", "status_code": 200} that contains additional conditions or restrictions on the request.

### USER QUESTION ###

What is number of apicalls for dates from 1 January 2021 to 20 Feb 2022
  """
  instruction = history
  kwargs = dict(temperature=0.6, top_p=0.9)
  if model_type == Model_Type.ggml:
      kwargs["max_tokens"] = 512
      for chunk in model(prompt=instruction, stream=True, **kwargs):
          token = chunk["choices"][0]["text"]
          history += token
          yield history

  else:
      streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
      inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
      kwargs["max_new_tokens"] = 512
      kwargs["input_ids"] = inputs["input_ids"]
      kwargs["streamer"] = streamer
      thread = Thread(target=model.generate, kwargs=kwargs)
      thread.start()

      for token in streamer:
          history += token
          yield history

      #msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
      #clear.click(lambda: None, None, chatbot, queue=False)
  #demo.queue()
  #demo.launch(share=True, debug=True)

def main(model_name=None, file_name=None):
    assert model_name is not None, "model_name argument is missing."

    is_chat_model = 'chat' in model_name.lower()
    model_type = get_model_type(model_name)

    if model_type == Model_Type.ggml:
      assert file_name is not None, "When model_name is provided for a GGML quantized model, file_name argument must also be provided."

    model, tokenizer = init_auto_model_and_tokenizer(model_name, model_type, file_name)
    for t in run_ui(model, tokenizer, is_chat_model, model_type):
        print(t)

if __name__ == '__main__':
  main(model_name="TheBloke/Llama-2-7B-Chat-GGML", file_name="llama-2-7b-chat.ggmlv3.q4_K_M.bin")