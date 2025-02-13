from gradio_helper import make_demo
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

if backend == "openvino":
  from vlmOv import model, processor
elif backend == "cuda":
  from vlmHf import model, processor
else:
  raise ValueError(f"Unknown backend: {backend}")

demo = make_demo(model, processor)

try:
  demo.launch(debug=True, height=600)
except Exception:
  demo.launch(debug=True, share=True, height=600)
