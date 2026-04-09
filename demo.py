import gradio as gr
from env import SupplyChainEnv

# Shared environment
env = SupplyChainEnv()
current_state = {}
logs = []

def reset_env(task_id):
    global current_state, logs
    obs = env.reset(task_id=task_id, seed=42)
    current_state = obs
    logs = [f"Environment reset: task={task_id}, initial observation={obs}"]
    return obs, "\n".join(logs)

def step_env(action_type, details_json):
    global current_state, logs
    try:
        # Build action dict like your API expects
        action = {
            "action_type": action_type,
            action_type: eval(details_json),  # assumes JSON-like dict as string
            "reasoning": "Manual input via demo"
        }
        response = env.step(action)
        current_state = response.observation
        logs.append(f"Step executed: {action_type}, result={response}")
        return current_state, "\n".join(logs)
    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return current_state, "\n".join(logs)

def grade_env():
    score = env.grade()
    logs.append(f"Episode finished. Score={score}")
    return score, "\n".join(logs)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Supply Chain Disruption Manager")
    with gr.Row():
        task_dropdown = gr.Dropdown(["supplier_triage", "logistics_reroute", "cascade_disruption"], label="Task")
        reset_btn = gr.Button("Reset")
    with gr.Row():
        action_dropdown = gr.Dropdown(["activate_supplier", "reroute_shipment", "allocate_stock", "negotiate_contract", "wait"], label="Action")
        details_text = gr.Textbox(label="Action details (dict as string)", placeholder='{"supplier_id": "SUP-001", "order_quantity": 100}')
        step_btn = gr.Button("Step")
    grade_btn = gr.Button("Finish Episode & Grade")
    output_state = gr.JSON(label="Current Observation")
    output_logs = gr.Textbox(label="Logs", lines=15)

    reset_btn.click(reset_env, inputs=task_dropdown, outputs=[output_state, output_logs])
    step_btn.click(step_env, inputs=[action_dropdown, details_text], outputs=[output_state, output_logs])
    grade_btn.click(grade_env, inputs=[], outputs=[output_state, output_logs])

demo.launch(server_name="0.0.0.0", server_port=7860)