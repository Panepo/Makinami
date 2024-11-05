from PIL import Image
from modelOV import model, processor
from transformers import TextStreamer
import cv2
import math

messages = [
    {"role": "user", "content": "<|image_1|>\nWhat is unusual on this picture?"},
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while(True):
    time1 = cv2.getTickCount()
    (grabbed, img) = cap.read()

    if not grabbed:
        break

    cv2.imshow("Webcam", img)

    cv2_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)

    inputs = processor(prompt, [pil_image], return_tensors="pt")
    generation_args = {"max_new_tokens": 50, "do_sample": False, "streamer": TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)}
    print("Answer:")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    time2 = cv2.getTickCount()
    time = math.floor( ((time2 - time1) * 1000) / cv2.getTickFrequency() )
    print(f"Elapsed time: {time} ms")

    getKey = cv2.waitKey(10) & 0xFF
    if getKey == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

