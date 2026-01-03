# llama158_chatbot.py

# 🧪 INSTALLATION (run this separately in terminal before launching)
# pip install torch --index-url https://download.pytorch.org/whl/cu121
# pip install git+https://github.com/huggingface/transformers.git@refs/pull/33410/head
# pip install gradio

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# 🧠 Load tokenizer and model
model_id = "HF1BitLLM/Llama3-8B-1.58-100B-tokens"
tokenizer_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

print("🧠 Loading 1.58-bit model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16  # Ensure GPU supports BF16 (e.g. A100/4090)
)

# 🗣️ Chat function
def chat(user_input, history):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assemble prompt from history
    full_input = ""
    for turn in history:
        full_input += f"User: {turn[0]}\nAssistant: {turn[1]}\n"
    full_input += f"User: {user_input}\nAssistant:"

    # Tokenize and truncate if needed
    input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=4000).to(device)
    model.to(device)

    try:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = response.split("Assistant:")[-1].strip()
    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"

    history.append((user_input, reply))
    return reply, history


# 🧙🏾‍♂️ Launch Gradio Chat Interface
with gr.Blocks(title="🦙 Llama3-8B-1.58 Chatbot") as demo:
    gr.Markdown("## 🦙 Llama3-8B-1.58 Chatbot\nChat with a super-efficient 1-bit model!")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message", placeholder="Ask me anything...")
    clear = gr.Button("Clear")

    state = gr.State([])

    def respond(user_message, history):
        reply, new_history = chat(user_message, history)
        return new_history, new_history

    msg.submit(respond, [msg, state], [chatbot, state])
    clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch(share=True,debug=True)
