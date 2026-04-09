import gradio as gr
from inference import run_inference

def run_demo():
    lines = run_inference()
    return "\n".join(lines)

with gr.Blocks() as demo:
    gr.Markdown("## Supply Chain Disruption Manager Demo")
    run_btn = gr.Button("Run Full Inference")
    output_box = gr.Textbox(label="Output", lines=30)

    run_btn.click(run_demo, outputs=output_box)

demo.launch(server_name="0.0.0.0", server_port=7860)