import os
import gradio as gr
from app import utils, clauses, navigator, drafting, rag_pipeline

# --- 1. SETUP & INITIALIZATION ---
os.environ['HF_HOME'] = 'D:/huggingface_cache'

try:
    # Use your custom model or a default like "mistral"
    rag_agent = rag_pipeline.RagPipeline(model="indian-legal-assistant") 
except Exception as e:
    rag_agent = None
    initialization_error = e

MAX_CLAUSES_TO_DISPLAY = 10

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
    if doc_state:
        rag_agent.create_knowledge_base(doc_state)
        answer = rag_agent.ask(message)
    else:
        answer = rag_agent.general_ask(message)
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
                    msg_box = gr.Textbox(placeholder="Ask a question about the document or a general legal question...", show_label=False, container=False)

        with gr.TabItem("🧭 Legal Navigator"):
            with gr.Row():
                with gr.Column(scale=1):
                    navigator_dropdown = gr.Dropdown(choices=list(navigator.LEGAL_GUIDES.keys()), label="Select a Legal Topic")
                    navigator_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=2):
                    navigator_chatbot = gr.Chatbot(height=500, label="Navigator Chat")
                    navigator_msg_box = gr.Textbox(show_label=False, container=False)
        
        # --- FIX START: Corrected Legal Drafting Assistant Tab ---
        with gr.TabItem("✍️ Legal Drafting Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Select Document Type")
                    drafting_dropdown = gr.Dropdown(
                        choices=list(drafting.DRAFTING_TEMPLATES.keys()),
                        label="Select Document Type"
                    )
                    
                    gr.Markdown("### 2. Fill in Details")
                    # Create and place all possible input fields here
                    drafting_inputs = {}
                    all_possible_fields = sorted(list({field for t in drafting.DRAFTING_TEMPLATES.values() for field in t["fields"]}))
                    
                    for field in all_possible_fields:
                        drafting_inputs[field] = gr.Textbox(label=field, visible=False)
                    
                    gr.Markdown("### 3. Generate")
                    draft_button = gr.Button("Generate Draft")

                with gr.Column(scale=2):
                    draft_output = gr.Textbox(label="Generated Draft", lines=35, interactive=True)
        # --- FIX END ---

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

    # --- FIX START: Corrected Event Listeners for Drafting ---
    def show_relevant_fields(template_name):
        required_fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {}).keys()
        updates = {}
        for field, component in drafting_inputs.items():
            updates[component] = gr.update(visible=(field in required_fields))
        return updates

    drafting_dropdown.change(
        fn=show_relevant_fields,
        inputs=[drafting_dropdown],
        outputs=list(drafting_inputs.values())
    )
    
    def generate_draft_ui(template_name, *args):
        details = {field: value for field, value in zip(all_possible_fields, args)}
        return drafting.generate_draft(template_name, details)

    draft_button.click(
        fn=generate_draft_ui,
        inputs=[drafting_dropdown] + list(drafting_inputs.values()),
        outputs=[draft_output]
    )
    # --- FIX END ---

# --- 5. App Launch ---
if __name__ == "__main__":
    if rag_agent is None:
        print("--- FATAL ERROR ---")
        print(f"Could not initialize the RAG agent. Please ensure Ollama is running.")
        print(f"Details: {initialization_error}")
    else:
        demo.launch(share=True)