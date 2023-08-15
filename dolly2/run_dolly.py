# pip install torch
# pip install transformers
# pip install accelerate
# pip install bitsandbytes
# pip install scipy

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from instruct_pipeline import InstructionTextGenerationPipeline

base_model = "databricks/dolly-v2-7b"
load_8bit = True

tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)

pipe = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)
pipe("Write an example of JSON file:")