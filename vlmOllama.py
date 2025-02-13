import ollama
from PIL import Image
import base64
import io

def image_to_base64(image_path):
  # Open the image file
  with Image.open(image_path) as img:
    # Create a BytesIO object to hold the image data
    buffered = io.BytesIO()
    # Save the image to the BytesIO object in a specific format (e.g., JPEG)
    img.save(buffered, format="PNG")
    # Get the byte data from the BytesIO object
    img_bytes = buffered.getvalue()
    # Encode the byte data to base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

import time
start_time = time.process_time()

image_path = 'small.png' # Replace with your image path
base64_image = image_to_base64(image_path)

response = ollama.chat(
  model='llama3.2-vision',
  messages=[
    {
      'role': 'user',
      'content': 'What is the text saying?',
      'images': [base64_image]
    }
  ]
)

cleaned_text = response['message']['content'].strip()
print(cleaned_text)

end_time = time.process_time()
elapsed_time = end_time - start_time
print(f"Process time: {elapsed_time} seconds")
