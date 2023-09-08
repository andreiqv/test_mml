# pip install torch
# pip install transformers
# pip install accelerate
# pip install bitsandbytes
# pip install scipy

import os
import time
from enum import Enum
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
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


    prompt="""
### INSTRUCTION:

Based on the user question below, generate a request to an external API as a JSON dictionary with the following keys:
- "question_type", string: possible values "apicall" (if it is about api calls), "other" (if about something else);
- "target_fields", a list of required fields about which the user asks, possible values: "apicall_uid", "session_uid", "testrun_requestid", "pageviews_project_id", "created_at" (date and time of creation), "updated_at" (date and time of updating), "is_active", "url", "originating_page_url", "method", "status" (like "200 - OK"), "status_code" (int, like 200), "response_type" (usually "str"), "response_time" (int value in seconds), "request_headers", "response_headers", "request_body", "response_body"; 
- "aggregations": a list of required aggregations for pandas aggregation function, possible values: "count", "nunique" (unique count), "max", "min", "sum", "average", "median", "mode", "std" (standard_deviation), "var" (variance), "quantile";
- "dates": a dict with a range of dates like {"start_date": "2023-07-26", "end_date": "2023-07-30"), or a specific date {"specific_date": "2023-08-01"} if the user defined it, overwise return the empty dict;
- "filter_params": a dict like {"method": "get", "status_code": 200} that contains additional conditions or restrictions on the request.

### USER QUESTION:

What's the highest score in the recent game?

    """
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