import spaces  # ZeroGPU ke liye ye hamesha sabse pehli line honi chahiye
import gradio as gr
from core import CommentLensAnalyzer, extract_video_id

analyzer = CommentLensAnalyzer()
HF_CACHE = {}


@spaces.GPU(duration=15)
def run_on_gpu(video_id: str, max_comments: int):
    return analyzer.analyze(video_id, max_comments)


def analyze_video(video_id: str, max_comments: int = 150):
    vid = extract_video_id(video_id)
    if not vid:
        return {"error": "Please provide a valid YouTube video URL or 11-character video ID."}

    max_comments = max(10, min(int(max_comments), 200))
    cache_key = f"v2:{vid}:{max_comments}"
    if cache_key in HF_CACHE:
        return HF_CACHE[cache_key]

    result = run_on_gpu(vid, max_comments)
    if isinstance(result, dict) and "error" not in result:
        HF_CACHE[cache_key] = result
    return result


with gr.Blocks(title="CommentLens API") as demo:
    gr.Markdown(
        "## CommentLens Sentiment API\n"
        "Paste any YouTube Video URL or ID to test live sentiment, keywords, and timeline analytics."
    )
    with gr.Row():
        vid_input = gr.Textbox(
            label="YouTube URL or Video ID",
            placeholder="e.g. https://youtube.com/shorts/Dc654YcgN4E",
        )
        max_input = gr.Slider(
            minimum=20, maximum=200, value=150, step=10, label="Max Comments"
        )
    run_btn = gr.Button("Analyze Comments", variant="primary")
    output_json = gr.JSON(label="Sentiment Analysis Result")

    run_btn.click(
        fn=analyze_video,
        inputs=[vid_input, max_input],
        outputs=output_json,
        api_name="analyze",
    )

if __name__ == "__main__":
    demo.launch()