from gradio_helper import make_demo
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

if backend == "openvino":
  from ovmodels import model, processor
elif backend == "cuda":
  from hfmodels import model, processor

demo = make_demo(model, processor)

try:
  demo.launch(debug=True, height=600)
except Exception:
  demo.launch(debug=True, share=True, height=600)
