import gradio as gr
import subprocess

def run_demo():
    result = subprocess.run(
        ["python", "inference.py"],
        capture_output=True,
        text=True
    )
    return result.stdout

with gr.Blocks() as demo:
    gr.Markdown("# 🚚 Supply Chain OpenEnv Demo")

    gr.Markdown("Click below to run an OpenEnv task")

    btn = gr.Button("Run Task")

    output = gr.Textbox(label="Execution Logs", lines=20)

    btn.click(fn=run_demo, inputs=[], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7861)