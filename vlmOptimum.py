from optimum.intel.openvino import OVModelForVisualCausalLM
from transformers import AutoProcessor, TextStreamer
from vlm_config import model_dir, model_path
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

target_dir = model_dir / model_path[backend]
processor = AutoProcessor.from_pretrained(target_dir, trust_remote_code=True)
model = OVModelForVisualCausalLM.from_pretrained(target_dir, trust_remote_code=True)
