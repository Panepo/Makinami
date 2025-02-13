from pathlib import Path
import gradio as gr
from transformers import TextIteratorStreamer
from threading import Thread
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

backend = os.getenv("BACKEND")

if backend == "openvino":
  from vlmOv import model, processor
elif backend == "cuda" or backend == "cpu":
  from vlmHf import model, processor
elif backend == "optimum":
  from vlmOptimum import model, processor
  raise NotImplementedError("Optimum backend is not supported")
else:
  raise ValueError(f"Unknown backend: {backend}")

def fn_camera(image):
  return image.transpose(Image.FLIP_LEFT_RIGHT)

def fn_llm(image, question):
  if image is None:
    return "Please upload an image"
  elif question is None:
    return "Please enter a question"
  else:
    messages = [ {"role": "user", "content": f"<|image_1|>\n{question}"},]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if backend == "openvino" or backend == "cpu":
      inputs = processor(prompt, [image], return_tensors="pt")
    elif backend == "cuda":
      inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    elif backend == "optimum":
      inputs = model.preprocess_inputs(text=prompt, image=image, processor=processor)

    streamer = TextIteratorStreamer(processor,
      **{
          "skip_special_tokens": True,
          "skip_prompt": True,
          "clean_up_tokenization_spaces": False,
      },
    )

    if backend == "openvino" or backend == "cpu" or backend == "cuda":
      generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=False,
        eos_token_id=processor.tokenizer.eos_token_id,
      )
    elif backend == "optimum":
      generation_kwargs = {
        "max_new_tokens": 50,
        "temperature": 0.0,
        "do_sample": False,
        "streamer": streamer,
      }

      generate_ids = model.generate(**inputs,
        eos_token_id=processor.tokenizer.eos_token_id,
        **generation_kwargs
      )

      generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
      response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
      return response

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    buffer = ""
    for new_text in streamer:
      buffer += new_text
      yield buffer


with gr.Blocks() as demo:
  with gr.Row():
    with gr.Column():
      img_input = gr.Image(label="camera", sources="webcam", streaming=True, type="pil")
    with gr.Column():
      txt_input = gr.Textbox(label="Ask a question", lines=2)
      btn_input = gr.Button("Submit")
      txt_output = gr.Textbox(label="LLM Anwsers", lines=2)

  img_input.stream(fn_camera, img_input, img_input, stream_every=0.5, concurrency_limit=30)
  txt_input.submit(fn_llm, [img_input, txt_input], txt_output, queue=True)
  btn_input.click(fn_llm, [img_input, txt_input], txt_output, queue=True)

if __name__ == "__main__":
  demo.launch()
