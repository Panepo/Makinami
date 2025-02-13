import argparse
from vlm_config import model_dir, model_path

def download_model(env):
  if not model_dir.exists():
    model_dir.mkdir()

  def git_clone(repo, path):
    import subprocess  # nosec - disable B404:import-subprocess check
    subprocess.run(["git", "clone", repo, path], check=True)

  target_dir = model_dir / model_path[env]

  if not target_dir.exists() and env == "optimum":
    print("Downloading Phi-3.5 vision model for optimum...")
    git_clone("https://huggingface.co/OpenVINO/Phi-3.5-vision-instruct-int4-ov", target_dir)

  if not target_dir.exists() and env == "cuda":
    print("Downloading Phi-3.5 vision model for cuda...")
    git_clone("https://huggingface.co/microsoft/Phi-3.5-vision-instruct", target_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-e",
    "--env",
    required=True,
    help="Download Phi3.5 vision model for different environment, currently supported cuda and openvino",
  )
  args = parser.parse_args()

  download_model(args.env)
