from huggingface_hub import HfApi, create_repo
import sys
import os

from dotenv import load_dotenv

# Load token
load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    print("Error: HF_TOKEN not found in .env file.")
    exit(1)

api = HfApi(token=token)

try:
    user_info = api.whoami()
    username = user_info['name']
    repo_id = f"{username}/mnist-cnn-trainer"
    print(f"Deploying to: {repo_id}")
    
    try:
        create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", private=False, token=token)
    except Exception as e:
        print(f"Space already exists or error: {e}")

    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
    )
    print(f"SUCCESS: Deployed to https://huggingface.co/spaces/{repo_id}")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
