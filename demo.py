import gradio as gr
from env import SupplyChainEnv

env = SupplyChainEnv()

def reset_task(task_id):
    obs = env.reset(task_id=task_id, seed=42)
    return obs

iface = gr.Interface(
    fn=reset_task,
    inputs=gr.Dropdown(["supplier_triage", "logistics_reroute", "cascade_disruption"]),
    outputs="json",
    title="Supply Chain Disruption Manager",
    description="Interact with your OpenEnv environment"
)

iface.launch(server_name="0.0.0.0", server_port=7860)