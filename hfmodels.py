from pathlib import Path
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

model_dir = Path("models") / Path("Phi-3.5-vision-instruct") / "FP16"

model = AutoModelForCausalLM.from_pretrained(
  model_dir,
  device_map="cuda",
  trust_remote_code=True,
  torch_dtype="auto",
  _attn_implementation='flash_attention_2'
)

processor = AutoProcessor.from_pretrained(
  model_dir,
  trust_remote_code=True,
  num_crops=4
)
