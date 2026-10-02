import os

import gradio as gr
import torch

from model import ChatModerator
from vectorizer import BigramVectorizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Load the trained model + fitted vocab once, at startup
# ---------------------------------------------------------------------------

# Trained on real, human-annotated Dota 2 chat (CONDA dataset,
# Weld et al. 2021 — https://github.com/usydnlp/CONDA),
vectorizer = BigramVectorizer()
vectorizer.load(os.path.join(SCRIPT_DIR, "vocab.json"))

model = ChatModerator(vocab_size=vectorizer.vocab_size, hidden_dim=64)
model.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, "model.pt"), map_location="cpu"))
model.eval()  # inference mode, not training


def classify(text):
    """Returns (is_toxic: bool, probability: float)."""
    vector = vectorizer.transform(text)
    x = torch.tensor([vector], dtype=torch.float32)  # shape (1, vocab_size)

    with torch.no_grad():
        logit = model(x)
        probability = torch.sigmoid(logit).item()  # only place sigmoid gets applied

    return probability > 0.5, probability



MAX_STRIKES = 3


def chat_moderate(message, history, strikes):
    is_toxic, probability = classify(message)

    if is_toxic:
        strikes += 1
        remaining = MAX_STRIKES - strikes
        if remaining > 0:
            reply = f"Message flagged as toxic ({probability:.0%} confidence). Strike {strikes}/{MAX_STRIKES}."
        else:
            reply = f"Message flagged as toxic ({probability:.0%} confidence). Strike {strikes}/{MAX_STRIKES} — you're banned."
    else:
        reply = f"Message looks fine ({1 -probability:.0%} confidence)."

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply})

    banned = strikes >= MAX_STRIKES
    # once banned: clear + lock the textbox so no further messages can be sent
    box_update = gr.update(value="", interactive=not banned, placeholder="You've been chat restricted XD." if banned else "Type something and hit enter...")
    return history, strikes, box_update


def reset_chat():
    return [], 0, gr.update(value="", interactive=True, placeholder="Type something and hit enter...")


# ---------------------------------------------------------------------------
# Build the interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Chat Moderator") as demo:
    gr.Markdown(
        "# Chat Moderator Demo\nSOCIS AI/ML Workshop — Session 1\n\n"
        "*Trained on the [CONDA dataset](https://github.com/usydnlp/CONDA) "
        "(Weld et al., 2021) — real, human-annotated Dota 2 chat logs.*"
    )

    strikes_state = gr.State(0)
    chatbot = gr.Chatbot(label="Simulated chat", type="messages")
    msg_box = gr.Textbox(label="Type a message", placeholder="Type something and hit enter...")
    clear_btn = gr.Button("Reset chat")

    msg_box.submit(
        chat_moderate,
        inputs=[msg_box, chatbot, strikes_state],
        outputs=[chatbot, strikes_state, msg_box],
    )
    clear_btn.click(reset_chat, outputs=[chatbot, strikes_state, msg_box])

if __name__ == "__main__":
    demo.launch()
