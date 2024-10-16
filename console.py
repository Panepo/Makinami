from PIL import Image
from transformers import TextStreamer
import time
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

if backend == "openvino":
  from ovmodels import model, processor
elif backend == "cuda":
  from hfmodels import model, processor

messages = [
    {"role": "user", "content": "<|image_1|>\nWhat is unusual on this picture?"},
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

start_time = time.process_time()
image = Image.open("./1.jpg")

if backend == "openvino":
  inputs = processor(prompt, [image], return_tensors="pt")
elif backend == "cuda":
  inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {"max_new_tokens": 50, "do_sample": False, "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}
print("Answer:")
generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
