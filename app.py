import os
import gradio as gr
from app import utils, rag_pipeline, clauses, translator, navigator, drafting
from langdetect import detect

# --- 1. SETUP & INITIALIZATION ---

os.environ['HF_HOME'] = 'D:/huggingface_cache'#change it according to your dir

try:
    rag_agent = rag_pipeline.RagPipeline()
except Exception as e:
    rag_agent = None
    initialization_error = e

LANG_CODE_MAPPING = {
    "bn": "bn_IN", "gu": "gu_IN", "hi": "hi_IN", "ml": "ml_IN",
    "mr": "mr_IN", "ta": "ta_IN", "te": "te_IN", "ur": "ur_PK", "en": "en_XX"
}
output_language_choices = ["English (en_XX)"] + [f"{name} ({code})" for code, name in translator.INDIAN_LANGUAGES.items()]
navigator_topic_choices = list(navigator.LEGAL_GUIDES.keys())
drafting_template_choices = list(drafting.DRAFTING_TEMPLATES.keys())
MAX_CLAUSES_TO_DISPLAY = 10 

# --- 2. CORE GRADIO FUNCTIONS ---

def process_file(file, progress=gr.Progress(track_tqdm=True)):
    """Loads a file, extracts text, builds the KB, and returns updates for the static UI."""
    if not file:
        # Create a list of updates to clear all relevant fields
        updates = ["", "", "Please upload a file first."]
        for _ in range(MAX_CLAUSES_TO_DISPLAY):
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
        return tuple(updates)

    progress(0, desc="Cleaning and extracting text...")
    extracted_text = utils.load_file_content(file.name)
    
    if "Error:" in extracted_text:
        updates = [extracted_text, extracted_text, extracted_text]
        for _ in range(MAX_CLAUSES_TO_DISPLAY):
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
        return tuple(updates)

    progress(0.5, desc="Building knowledge base...")
    rag_agent.create_knowledge_base(extracted_text)
    
    progress(0.8, desc="Scanning for clauses...")
    found_clauses = clauses.extract_clauses(extracted_text, rag_agent)
    
    # --- CRITICAL STATE MANAGEMENT FIX ---
    # The raw extracted_text string MUST be returned to correctly update the doc_state
    updates = [extracted_text[:3000], extracted_text, "Document processed successfully."]
    for i in range(MAX_CLAUSES_TO_DISPLAY):
        if i < len(found_clauses):
            clause = found_clauses[i]
            updates.append(gr.update(label=f"Clause Found: {clause['title']} (Risk: {clause['risk']})", visible=True))
            updates.append(gr.update(value=f"**Extracted Text:**\n\n> {clause['text']}", visible=True))
            updates.append(gr.update(visible=True))
        else:
            updates.extend([gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)])
            
    return tuple(updates)

def explain_clause(clause_text):
    """Uses the AI to explain a single clause."""
    try:
        if not clause_text or ">" not in clause_text:
            return gr.update(value="Could not find text to explain.", visible=True)
        core_text = clause_text.split(">")[-1].strip()
        prompt = f"Please explain the following legal clause in simple, easy-to-understand terms:\n\n---\n{core_text}\n---"
        explanation = rag_agent.ask(prompt)
        return gr.update(value=explanation, visible=True)
    except Exception as e:
        return gr.update(value=f"Error explaining clause: {e}", visible=True)

def get_summary(doc_state, progress=gr.Progress(track_tqdm=True)):
    if not isinstance(doc_state, str) or not doc_state: return "Please upload a document first."
    try:
        progress(0, desc="Generating summary..."); return rag_agent.summarize(doc_state)
    except Exception as e: return f"Error during summary: {e}"

def respond(message, history, doc_state, output_language_selection):
    """Handles document Q&A chat, always returning two values."""
    if not isinstance(doc_state, str) or not doc_state:
        history.append((message, "Please upload and process a document first."))
        return "", history

    try:
        target_lang_code = output_language_selection.split('(')[-1].replace(')', '').strip()
        try: source_lang_code = LANG_CODE_MAPPING.get(detect(message), "en_XX")
        except: source_lang_code = "en_XX"
            
        english_query = translator.translate_text(message, src_lang=source_lang_code, target_lang="en_XX") if source_lang_code != "en_XX" else message
        english_answer = rag_agent.ask(english_query)
        final_answer = translator.translate_text(english_answer, src_lang="en_XX", target_lang=target_lang_code) if target_lang_code != "en_XX" else english_answer
        
        history.append((message, final_answer))
        return "", history
    except Exception as e:
        history.append((message, f"An error occurred: {e}"))
        return "", history

def load_guide(topic):
    guide_text = navigator.LEGAL_GUIDES.get(topic);
    if guide_text:
        rag_agent.create_knowledge_base(guide_text); return f"Guide for '{topic}' loaded.", []
    return "Failed to load guide.", []

def navigator_respond(message, history):
    try:
        response = rag_agent.ask(message); history.append((message, response)); return "", history
    except Exception as e:
        history.append((message, f"An error occurred: {e}")); return "", history

def generate_legal_draft(template_name, *args):
    fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {});
    details = {label: arg for label, arg in zip(fields.keys(), args)};
    return drafting.generate_draft(template_name, details)

# --- GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="LegiWise AI") as demo:
    doc_state = gr.State() # Initialize the state object
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
                        msg_box = gr.Textbox(label="Ask a question...", show_label=False, container=False, scale=3)

        with gr.TabItem("🧭 Legal Navigator"):
            with gr.Row():
                with gr.Column(scale=1):
                    navigator_dropdown = gr.Dropdown(choices=navigator_topic_choices, label="Select a Legal Topic")
                    navigator_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=2):
                    navigator_chatbot = gr.Chatbot(height=500, label="Navigator Chat")
                    navigator_msg_box = gr.Textbox(show_label=False, container=False)
        
        with gr.TabItem("✍️ Legal Drafting Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    drafting_dropdown = gr.Dropdown(choices=drafting_template_choices, label="Select Document Type")
                    draft_button = gr.Button("Generate Draft")
                    drafting_inputs = {}
                    unique_fields = {field for t in drafting.DRAFTING_TEMPLATES.values() for field in t["fields"].keys()}
                    for field in unique_fields:
                        drafting_inputs[field] = gr.Textbox(label=field, visible=False)
                with gr.Column(scale=2):
                    draft_output = gr.Textbox(label="Generated Draft", lines=25, interactive=True)

        with gr.TabItem("🔍 Extracted Text & Clauses"):
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

    # --- Event Listeners ---
    # CRITICAL FIX: The process_file function now has 3 + (3 * MAX_CLAUSES) outputs.
    # The second output is the raw text, which correctly updates the doc_state.
    file_input.upload(
        process_file, 
        [file_input], 
        [text_preview_output, doc_state, status_box] + clause_outputs
    )
    summary_button.click(get_summary, [doc_state], [summary_output])
    msg_box.submit(respond, [msg_box, chatbot, doc_state, output_language_dropdown], [msg_box, chatbot])
    
    navigator_dropdown.change(load_guide, [navigator_dropdown], [navigator_status, navigator_chatbot])
    navigator_msg_box.submit(navigator_respond, [navigator_msg_box, navigator_chatbot], [navigator_msg_box, navigator_chatbot])
    
    def show_relevant_fields(template_name):
        template_fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {}).keys()
        updates = {}
        for field, component in drafting_inputs.items():
            updates[component] = gr.update(visible=field in template_fields)
        return updates
    
    all_field_components = list(drafting_inputs.values())
    drafting_dropdown.change(show_relevant_fields, [drafting_dropdown], all_field_components)
    draft_button.click(generate_legal_draft, [drafting_dropdown] + all_field_components, [draft_output])

if __name__ == "__main__":
    if rag_agent is None:
        print(f"FATAL ERROR: Could not initialize RAG agent. Error: {initialization_error}")
    else:
        demo.launch(share=True)

