import threading
import gradio as gr
import requests
import subprocess

# Start FastAPI server in background
def start_api():
    subprocess.Popen(["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"])

threading.Thread(target=start_api, daemon=True).start()

# Wait a few seconds for API to start
import time
time.sleep(3)

# Example Gradio interface
def reset_task(task_id):
    r = requests.post("http://localhost:7860/reset", json={"task_id": task_id})
    return r.json()

iface = gr.Interface(
    fn=reset_task,
    inputs=gr.Dropdown(["supplier_triage", "logistics_reroute", "cascade_disruption"]),
    outputs="json",
    title="Supply Chain Disruption Manager",
    description="Interact with your OpenEnv environment"
)

iface.launch(server_name="0.0.0.0", server_port=7860)