from ov_phi3_vision_helper import OvPhi3Vision
from transformers import AutoProcessor
from vlm_config import model_dir, model_path
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

target_dir = model_dir / model_path[backend]
model = OvPhi3Vision(target_dir, "GPU")
processor = AutoProcessor.from_pretrained(target_dir, trust_remote_code=True)
