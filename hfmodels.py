from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

model_dir = "./models/Phi-3.5-vision-instruct/FP16"

if backend == "cuda":
  model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
  )
elif backend == "cpu":
  model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
  )

processor = AutoProcessor.from_pretrained(
  model_dir,
  trust_remote_code=True,
  num_crops=4
)
