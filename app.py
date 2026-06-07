"""Build Small Hackathon — Gradio app entrypoint.

Constraints (see project memory):
  - Model total params <= 32B (Tiny Titan special award is <= 4B).
  - Must be a Gradio app hosted as a Hugging Face Space.
"""

import gradio as gr


def respond(message: str) -> str:
    # TODO: wire up the small model here.
    return f"You said: {message}"


with gr.Blocks(title="hackaton-small") as demo:
    gr.Markdown("# 🏔️ Build Small Hackathon\nSmall models, big adventure.")
    inp = gr.Textbox(label="Say something")
    out = gr.Textbox(label="Response")
    inp.submit(respond, inputs=inp, outputs=out)


if __name__ == "__main__":
    demo.launch()
