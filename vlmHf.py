from transformers import AutoModelForCausalLM, AutoProcessor
from vlm_config import model_dir, model_path
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")
bit4 = True if os.getenv("BIT4") == 'true' else False

target_dir = model_dir / model_path[backend]

# Note: set _attn_implementation='eager' if you don't have flash_attn installed
if backend == "cuda":
  model = AutoModelForCausalLM.from_pretrained(
    target_dir,
    device_map="cuda",
    load_in_4bit=bit4,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
  )
elif backend == "cpu":
  model = AutoModelForCausalLM.from_pretrained(
    target_dir,
    load_in_4bit=bit4,
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation='flash_attention_2'
  )

processor = AutoProcessor.from_pretrained(
  target_dir,
  trust_remote_code=True,
  num_crops=4
)
