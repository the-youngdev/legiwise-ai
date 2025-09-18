import os
import gradio as gr
from app import utils, clauses, translator, navigator, drafting
from app.rag_pipeline import RagPipeline

# --- 1. SETUP & INITIALIZATION ---
#os.environ['HF_HOME'] = 'D:/huggingface_cache'

try:
    rag_agent = RagPipeline(model="mistral")
except Exception as e:
    # This will catch errors during initialization, like the Ollama connection issue
    rag_agent = None
    initialization_error = e

MAX_CLAUSES_TO_DISPLAY = 10
output_language_choices = ["English (en_XX)"] + [
    f"{name} ({code})" for name, code in translator.INDIAN_LANGUAGES.items()
]

# --- 2. CORE GRADIO FUNCTIONS ---
def process_file(file, progress=gr.Progress(track_tqdm=True)):
    if not file:
        updates = ["", "", "Please upload a file first."] + [gr.update(visible=False)] * (MAX_CLAUSES_TO_DISPLAY * 3)
        return tuple(updates)
    progress(0.2, desc="Extracting text...")
    text = utils.load_file_content(file.name)
    if "Error:" in text:
        updates = [text, text, text] + [gr.update(visible=False)] * (MAX_CLAUSES_TO_DISPLAY * 3)
        return tuple(updates)
    progress(0.5, desc="Building knowledge base...")
    rag_agent.create_knowledge_base(text)
    progress(0.8, desc="Scanning for clauses...")
    found_clauses = clauses.extract_clauses(text)
    updates = [text[:4000], text, "Document processed successfully."]
    for i in range(MAX_CLAUSES_TO_DISPLAY):
        if i < len(found_clauses):
            clause = found_clauses[i]
            updates.extend([
                gr.update(label=f"Clause: {clause['title']} (Risk: {clause['risk']})", visible=True),
                gr.update(value=f"**Extracted Text:**\n> {clause['text']}", visible=True),
                gr.update(visible=True)
            ])
        else:
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
    return tuple(updates)

def explain_clause(clause_text):
    core_text = clause_text.split(">")[-1].strip()
    prompt = f"Please explain this legal clause in simple terms:\n\n---\n{core_text}\n---"
    return rag_agent.ask(prompt)

def get_summary(doc_state, progress=gr.Progress(track_tqdm=True)):
    if not doc_state:
        return "Please upload a document first."
    progress(0, desc="Generating summary...")
    return rag_agent.summarize(doc_state)

def respond(message, history, doc_state):
    if not doc_state:
        history.append((message, "Please upload a document first."))
        return "", history
    rag_agent.create_knowledge_base(doc_state)
    answer = rag_agent.ask(message)
    history.append((message, answer))
    return "", history

def load_guide(topic):
    guide = navigator.LEGAL_GUIDES.get(topic)
    if guide:
        rag_agent.create_knowledge_base(guide)
        return f"Guide for '{topic}' loaded. Ask questions below.", [], guide
    return "Failed to load guide.", [], ""

def navigator_respond(message, history):
    response = rag_agent.ask(message)
    history.append((message, response))
    return "", history

# --- 3. GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="LegiWise AI") as demo:
    doc_state = gr.State("")
    gr.Markdown("# ⚖️ LegiWise AI – Your Personal Legal Assistant")

    with gr.Tabs():
        with gr.TabItem("📄 Document Analysis & Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Upload Legal Document")
                    status_box = gr.Textbox(label="Status", interactive=False)
                    summary_button = gr.Button("📄 Generate Summary")
                    summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Document Chat", height=500)
                    with gr.Row():
                        output_language_dropdown = gr.Dropdown(choices=output_language_choices, value="English (en_XX)", label="Answer Language", min_width=150)
                        msg_box = gr.Textbox(placeholder="Ask a question about the document...", show_label=False, container=False, scale=3)

        with gr.TabItem("🧭 Legal Navigator"):
            with gr.Row():
                with gr.Column(scale=1):
                    navigator_dropdown = gr.Dropdown(choices=list(navigator.LEGAL_GUIDES.keys()), label="Select a Legal Topic")
                    navigator_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=2):
                    navigator_chatbot = gr.Chatbot(height=500, label="Navigator Chat")
                    navigator_msg_box = gr.Textbox(show_label=False, container=False)

        with gr.TabItem("✍️ Legal Drafting Assistant"):
            # This section is filled out for completeness
            with gr.Row():
                with gr.Column(scale=1):
                    drafting_dropdown = gr.Dropdown(choices=list(drafting.DRAFTING_TEMPLATES.keys()), label="Select Document Type")
                    draft_button = gr.Button("Generate Draft")
                with gr.Column(scale=2):
                    draft_output = gr.Textbox(label="Generated Draft", lines=25, interactive=True)

        with gr.TabItem("🔍 Extracted Clauses"):
            with gr.Row():
                text_preview_output = gr.Textbox(label="Extracted Text Preview", lines=25, interactive=False)
                with gr.Column():
                    gr.Markdown("### Clause Analysis")
                    clause_outputs = []
                    for i in range(MAX_CLAUSES_TO_DISPLAY):
                        with gr.Accordion(f"Clause #{i+1}", visible=False) as acc:
                            content = gr.Markdown(visible=False)
                            explain_button = gr.Button("🤖 Explain this Clause with AI", visible=False)
                            explanation_box = gr.Textbox(label="AI Explanation", interactive=False, visible=False)
                            explain_button.click(explain_clause, inputs=[content], outputs=[explanation_box])
                            clause_outputs.extend([acc, content, explain_button])

    # --- 4. Event Listeners ---
    file_input.upload(
        process_file,
        [file_input],
        [text_preview_output, doc_state, status_box] + clause_outputs
    )
    summary_button.click(get_summary, [doc_state], [summary_output])
    msg_box.submit(respond, [msg_box, chatbot, doc_state], [msg_box, chatbot])
    
    navigator_dropdown.change(load_guide, [navigator_dropdown], [navigator_status, navigator_chatbot, doc_state])
    navigator_msg_box.submit(navigator_respond, [navigator_msg_box, navigator_chatbot], [navigator_msg_box, navigator_chatbot])

# --- 5. App Launch ---
if __name__ == "__main__":
    if rag_agent is None:
        print("--- FATAL ERROR ---")
        print(f"Could not initialize the RAG agent. Please ensure Ollama is running.")
        print(f"Details: {initialization_error}")
    else:
        demo.launch(share=True)
