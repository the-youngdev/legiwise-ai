import os
import gradio as gr
from app import utils, clauses, navigator, drafting, rag_pipeline

# --- 1. SETUP ---
os.environ['HF_HOME'] = os.environ.get("HF_HOME", "/home/ubuntu/.cache/huggingface")

try:
    rag_agent = rag_pipeline.RagPipeline(model="indian-legal-assistant")
except Exception as e:
    rag_agent = None
    initialization_error = e

MAX_CLAUSES_TO_DISPLAY = 10


# --- 2. CORE FUNCTIONS ---
def process_file(file, progress=gr.Progress(track_tqdm=True)):
    if not file:
        return "⚠️ Please upload a file first.", "", "⚠️", *[gr.update(visible=False)] * (MAX_CLAUSES_TO_DISPLAY * 3)
    progress(0.2, desc="Extracting text...")
    text = utils.load_file_content(file.name)
    if "Error:" in text:
        return text, text, text, *[gr.update(visible=False)] * (MAX_CLAUSES_TO_DISPLAY * 3)
    progress(0.5, desc="Building knowledge base...")
    rag_agent.create_knowledge_base(text)
    progress(0.8, desc="Scanning for clauses...")
    found_clauses = clauses.extract_clauses(text)
    updates = [text[:4000], text, "✅ Document processed successfully."]
    for i in range(MAX_CLAUSES_TO_DISPLAY):
        if i < len(found_clauses):
            clause = found_clauses[i]
            updates.extend([
                gr.update(label=f"{clause['title']} (Risk: {clause['risk']})", visible=True),
                gr.update(value=f"**Extracted Text:**\n> {clause['text']}", visible=True),
                gr.update(visible=True)
            ])
        else:
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
    return tuple(updates)


def explain_clause(clause_text):
    core_text = clause_text.split(">")[-1].strip()
    return rag_agent.ask(f"Explain this legal clause in simple terms:\n\n{core_text}")


def get_summary(doc_state, progress=gr.Progress(track_tqdm=True)):
    if not doc_state:
        return "⚠️ Please upload a document first."
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
        return f"Guide for '{topic}' loaded.", [], guide
    return "⚠️ Failed to load guide.", [], ""


def navigator_respond(message, history):
    response = rag_agent.ask(message)
    history.append((message, response))
    return "", history


# --- 3. UI WITH CLEAN BUSINESS SIDEBAR ---
with gr.Blocks(
    title="⚖️ LegiWise AI",
    css="""
/* --- GLOBAL --- */
body { background: #343541; color: #ececec; font-family: 'Inter', sans-serif; }

/* Sidebar */
.sidebar { background: #202123; color: white; height: 100vh; padding: 12px; transition: width 0.3s ease; }
.sidebar-title { font-size: 18px !important; font-weight: 600 !important; color: #ececec !important; margin-bottom: 16px !important; }

/* Sidebar nav buttons */
.nav-btn {
  background: transparent !important;
  border: none !important;
  color: #d1d5db !important;
  text-align: left !important;
  font-size: 15px !important;
  padding: 10px 14px !important;
  border-radius: 6px !important;
  width: 100% !important;
  justify-content: flex-start !important;
}
.nav-btn:hover { background: #2a2b32 !important; color: white !important; }
.nav-btn.active { background: #343541 !important; color: #ffffff !important; font-weight: 600 !important; }

/* Main Panel */
.main-panel { padding: 20px; }

/* Cards */
.card {
  background: #2a2b32; border-radius: 12px; padding: 20px;
  box-shadow: 0 4px 6px rgba(0,0,0,0.2); margin-bottom: 20px;
  border: 1px solid #3a3b42;
}

/* Chatbot */
.gr-chatbot { 
  background: #40414f !important; 
  border-radius: 12px; border: 1px solid #565869; 
  color: #ececec !important;
}
"""
) as demo:
    doc_state = gr.State("")
    current_page = gr.State("analysis")

    with gr.Row():
        # --- Sidebar ---
        with gr.Column(scale=1, elem_classes="sidebar", min_width=200) as sidebar:
            gr.Markdown("## LegiWise AI", elem_classes="sidebar-title")

            nav_analysis = gr.Button("Document Analysis", elem_classes="nav-btn")
            nav_navigator = gr.Button("Legal Navigator", elem_classes="nav-btn")
            nav_drafting = gr.Button("Drafting Assistant", elem_classes="nav-btn")
            nav_clauses = gr.Button("Extracted Clauses", elem_classes="nav-btn")

        # --- Main Panel ---
        with gr.Column(scale=4, elem_classes="main-panel"):
            with gr.Group(visible=True) as page_analysis:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card"):
                        file_input = gr.File(label="Upload Document")
                        status_box = gr.Textbox(label="Status", interactive=False)
                        summary_button = gr.Button("📄 Generate Summary")
                        summary_output = gr.Textbox(label="Summary", lines=8, interactive=False)
                    with gr.Column(scale=2, elem_classes="card"):
                        chatbot = gr.Chatbot(label="Document Chat", height=450)
                        msg_box = gr.Textbox(placeholder="Ask about this document...", show_label=False)

            with gr.Group(visible=False) as page_navigator:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card"):
                        navigator_dropdown = gr.Dropdown(choices=list(navigator.LEGAL_GUIDES.keys()), label="Topic")
                        navigator_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=2, elem_classes="card"):
                        navigator_chatbot = gr.Chatbot(height=450, label="Navigator Chat")
                        navigator_msg_box = gr.Textbox(show_label=False, placeholder="Ask about this topic...")

            with gr.Group(visible=False) as page_drafting:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card"):
                        drafting_dropdown = gr.Dropdown(
                            choices=list(drafting.DRAFTING_TEMPLATES.keys()), label="Document Type"
                        )
                        drafting_inputs = {}
                        all_possible_fields = sorted(
                            list({field for t in drafting.DRAFTING_TEMPLATES.values() for field in t["fields"]})
                        )
                        for field in all_possible_fields:
                            drafting_inputs[field] = gr.Textbox(label=field, visible=False)
                        draft_button = gr.Button("Generate Draft")
                    with gr.Column(scale=2, elem_classes="card"):
                        draft_output = gr.Textbox(label="Generated Draft", lines=25, interactive=True)

            with gr.Group(visible=False) as page_clauses:
                with gr.Row():
                    text_preview_output = gr.Textbox(label="Extracted Text Preview", lines=20, interactive=False)
                    with gr.Column(scale=2, elem_classes="card"):
                        clause_outputs = []
                        for i in range(MAX_CLAUSES_TO_DISPLAY):
                            with gr.Accordion(f"Clause #{i+1}", visible=False) as acc:
                                content = gr.Markdown(visible=False)
                                explain_button = gr.Button("Explain", visible=False)
                                explanation_box = gr.Textbox(label="AI Explanation", interactive=False, visible=False)
                                explain_button.click(explain_clause, [content], [explanation_box])
                                clause_outputs.extend([acc, content, explain_button])

    # --- Event Bindings ---
    file_input.upload(process_file, [file_input], [text_preview_output, doc_state, status_box] + clause_outputs)
    summary_button.click(get_summary, [doc_state], [summary_output])
    msg_box.submit(respond, [msg_box, chatbot, doc_state], [msg_box, chatbot])
    navigator_dropdown.change(load_guide, [navigator_dropdown], [navigator_status, navigator_chatbot, doc_state])
    navigator_msg_box.submit(navigator_respond, [navigator_msg_box, navigator_chatbot], [navigator_msg_box, navigator_chatbot])

    # Drafting handlers
    def show_relevant_fields(template_name):
        required_fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {}).keys()
        return [gr.update(visible=(field in required_fields)) for field in drafting_inputs.keys()]

    drafting_dropdown.change(show_relevant_fields, [drafting_dropdown], list(drafting_inputs.values()))

    def generate_draft_ui(template_name, *args):
        details = {field: value for field, value in zip(all_possible_fields, args)}
        return drafting.generate_draft(template_name, details)

    draft_button.click(generate_draft_ui, [drafting_dropdown] + list(drafting_inputs.values()), [draft_output])

    # --- Sidebar Navigation ---
    def switch_page(page):
        return (
            gr.update(visible=(page == "analysis")),
            gr.update(visible=(page == "navigator")),
            gr.update(visible=(page == "drafting")),
            gr.update(visible=(page == "clauses")),
        )

    nav_analysis.click(lambda: switch_page("analysis"), None, [page_analysis, page_navigator, page_drafting, page_clauses])
    nav_navigator.click(lambda: switch_page("navigator"), None, [page_analysis, page_navigator, page_drafting, page_clauses])
    nav_drafting.click(lambda: switch_page("drafting"), None, [page_analysis, page_navigator, page_drafting, page_clauses])
    nav_clauses.click(lambda: switch_page("clauses"), None, [page_analysis, page_navigator, page_drafting, page_clauses])

# --- 4. Launch ---
if __name__ == "__main__":
    if rag_agent is None:
        print("❌ Could not initialize the RAG agent. Please ensure Ollama is running.")
        print(f"Details: {initialization_error}")
    else:
        demo.launch(share=True)
