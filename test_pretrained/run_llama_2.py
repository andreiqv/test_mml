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

#model_id = "TheBloke/Llama-2-13B-chat-GGML"

#from transformers import AutoTokenizer, AutoModelForCausalLM
#import torch
#from instruct_pipeline import InstructionTextGenerationPipeline

model_id = "TheBloke/Llama-2-7B-Chat-GGML"

from ctransformers import AutoModelForCausalLM

config = {'max_new_tokens': 256, 'repetition_penalty': 1.1, 'temperature': 0.1, 'stream': True}

llm = AutoModelForCausalLM.from_pretrained(model_id,
                                           model_type="llama",
                                           #lib='avx2', for cpu use
                                           gpu_layers=130, #110 for 7b, 130 for 13b
                                           **config
                                           )
     



if __name__ == '__main__':


    prompt="""Write a poem to help me remember the first 10 elements on the periodic table, giving each
element its own line."""
    tokens = llm.tokenize(prompt)

    result = llm(prompt, stream=False)
    print(result)

    while True:
        q = input("\nQuestion:")
        if len(q) < 3:
            break

        t0 = time.time()
        result = llm(q, stream=False)
        print("OUTPUT:", result)
        print("time: {:.3f} sec".format(time.time() - t0))
        print()