import gradio as gr
from inference import run_inference  # your existing inference.py should have a callable function

logs = []

def start_episode():
    """Run inference reset/start"""
    output = run_inference("reset")
    logs.append(f"✅ Reset: {output}")
    return output, "\n".join(logs)

def take_step():
    """Run one step using inference.py"""
    output = run_inference("step")
    logs.append(f"➡ Step output: {output}")
    return output, "\n".join(logs)

def finish_episode():
    """Finish episode and show final result"""
    output = run_inference("grade")
    logs.append(f"🏁 Episode finished: {output}")
    return output, "\n".join(logs)

with gr.Blocks() as demo:
    gr.Markdown("## Supply Chain Disruption Manager Demo (Automatic Inference Output)")

    start_btn = gr.Button("Start Episode")
    step_btn = gr.Button("Take Step")
    grade_btn = gr.Button("Finish Episode")

    output_box = gr.JSON(label="Current Output")
    logs_box = gr.Textbox(label="Logs", lines=15)

    start_btn.click(start_episode, outputs=[output_box, logs_box])
    step_btn.click(take_step, outputs=[output_box, logs_box])
    grade_btn.click(finish_episode, outputs=[output_box, logs_box])

demo.launch(server_name="0.0.0.0", server_port=7860)