import modal
import os
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",
        "torch",
        "transformers",
        "sentencepiece",
        "protobuf",
        "numpy",
        "requests",
    )
    .env({"HF_HOME": "/model_cache"})
    .run_commands(
        f'python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; '
        f'AutoTokenizer.from_pretrained(\'{MODEL_NAME}\'); '
        f'AutoModelForSequenceClassification.from_pretrained(\'{MODEL_NAME}\')"'
    )
    .env({
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TORCH_THREADS": "4",
    })
    .add_local_python_source("core")
)

app = modal.App("commentlens-api")

@app.function(secrets=[modal.Secret.from_name("commentlens-secrets")]) 
def some_function():
    os.getenv("YOUTUBE_API_KEY")


@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("commentlens-secrets")],
    scaledown_window=180,
    max_containers=2,
)
class CommentLensService:
    @modal.enter()
    def load_model(self):
        from core import CommentLensAnalyzer
        self.analyzer = CommentLensAnalyzer()

    @modal.fastapi_endpoint(method="POST")
    def analyze(self, payload: dict):
        video_id = payload.get("video_id", "")
        max_comments = payload.get("max_comments", 150)
        return self.analyzer.analyze(video_id, max_comments)