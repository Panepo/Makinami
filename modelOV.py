from pathlib import Path
from ov_phi3_vision_helper import OvPhi3Vision
from transformers import AutoProcessor

model_dir = "./models/Phi-3.5-vision-instruct/INT4"
model = OvPhi3Vision(model_dir, "GPU")
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
