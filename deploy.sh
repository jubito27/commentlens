#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/backend"

echo "1/2 Deploying to Modal..."
modal deploy modal_app.py

echo "2/2 Syncing core.py, app.py, requirements.txt to Hugging Face Space..."
python - <<'EOF'
import socket
from huggingface_hub import HfApi

# Force IPv4 to prevent Errno 104 TLS resets over IPv6
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_only

api = HfApi()
api.upload_folder(
    folder_path=".",
    repo_id="Abhishek2775/commentlens-api",
    repo_type="space",
    allow_patterns=["core.py", "app.py", "requirements.txt"],
    commit_message="Sync backend from deploy.sh",
)
print("HF Space synced successfully.")
EOF

echo "Done! Both Modal and Hugging Face Space updated."