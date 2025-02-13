import argparse
import sys

def is_venv():
  return (hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

def pip_install(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "install", *cli_args], check=True)

def pip_uninstall(*args):
  import subprocess  # nosec - disable B404:import-subprocess check

  cli_args = []
  for arg in args:
    cli_args.extend(str(arg).split(" "))
  subprocess.run([sys.executable, "-m", "pip", "uninstall", *cli_args], check=True)

def download_model(env):
  pip_install("-Uq", "pip")
  pip_install("python-dotenv", "transformers>=4.40", "gradio>=4.26")
  pip_install("protobuf>=3.20", "Pillow", "accelerate", "tqdm", "nncf>=2.11.0", "Requests", "numpy", "flash_attn")

  if env == "openvino":
    pip_uninstall("openvino", "openvino-tokenizers", "openvino_genai")
    pip_install("openvino", "openvino-tokenizers", "openvino_genai")

    pip_uninstall("torch", "torchvision")
    pip_install("--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch" "torchvision")
  elif env == "openvino-nightly":
    pip_uninstall("openvino", "openvino-tokenizers", "openvino_genai")
    pip_install("--pre", "--extra-index-url", "https://storage.openvinotoolkit.org/simple/wheels/nightly", "openvino", "openvino-tokenizers", "openvino_genai")

    pip_uninstall("torch", "torchvision")
    pip_install("--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch" "torchvision")
  else:
    pip_uninstall("openvino", "openvino-tokenizers", "openvino_genai")

    pip_uninstall("torch", "torchvision")
    pip_install("--extra-index-url", "https://download.pytorch.org/whl/cu124", "torch" "torchvision")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-e",
    "--env",
    required=True,
    help="Environment to install the dependencies",
  )
  args = parser.parse_args()

  if is_venv():
    download_model(args.env)
  else:
    print("Not running inside a virtual environment")
