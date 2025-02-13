from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("BACKEND")

from pathlib import Path

model_dir = Path("models")

model_path = {
  "openvino": "Phi-3.5-vision-instruct-int4-ov",
  "cuda": "Phi-3.5-vision-instruct",
  "cpu": "Phi-3.5-vision-instruct"
}
