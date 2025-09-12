import os
import gradio as gr
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, asdict
import json
from app import utils, rag_pipeline, clauses, translator, navigator, drafting
from langdetect import detect

# --- 1. ENHANCED STATE MANAGEMENT ---

@dataclass
class DocumentState:
    """Centralized state management for document processing."""
    raw_text: str = ""
    processed_text: str = ""
    file_name: str = ""
    file_path: str = ""
    quality_score: float = 0.0
    quality_level: str = "Unknown"
    has_knowledge_base: bool = False
    extracted_clauses: List[Dict] = None
    processing_status: str = "No document loaded"
    error_message: str = ""
    
    def __post_init__(self):
        if self.extracted_clauses is None:
            self.extracted_clauses = []
    
    def reset(self):
        """Reset all state to initial values."""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def is_ready_for_qa(self) -> bool:
        """Check if document is ready for Q&A."""
        return bool(self.processed_text and self.has_knowledge_base and not self.error_message)
    
    def get_status_message(self) -> str:
        """Get formatted status message."""
        if self.error_message:
            return f"❌ Error: {self.error_message}"
        elif not self.raw_text:
            return "📄 No document uploaded"
        elif not self.has_knowledge_base:
            return "⏳ Processing document..."
        else:
            quality_emoji = "🟢" if self.quality_score > 0.7 else "🟡" if self.quality_score > 0.4 else "🔴"
            return f"✅ Document ready | Quality: {quality_emoji} {self.quality_level} ({self.quality_score:.2f}) | Clauses: {len(self.extracted_clauses)}"

@dataclass
class AppState:
    """Global application state."""
    document: DocumentState
    rag_agent: Optional[Any] = None
    initialization_error: Optional[str] = None
    current_tab: str = "document_analysis"
    
    def __post_init__(self):
        if self.document is None:
            self.document = DocumentState()

# --- 2. SETUP & INITIALIZATION ---

os.environ['HF_HOME'] = 'D:/huggingface_cache'  # change it according to your dir

# Initialize global app state
app_state = AppState(document=DocumentState())

try:
    # Use enhanced RAG pipeline with quality awareness
    app_state.rag_agent = rag_pipeline.EnhancedRagPipeline()
except Exception as e:
    app_state.rag_agent = None
    app_state.initialization_error = str(e)

LANG_CODE_MAPPING = {
    "bn": "bn_IN", "gu": "gu_IN", "hi": "hi_IN", "ml": "ml_IN",
    "mr": "mr_IN", "ta": "ta_IN", "te": "te_IN", "ur": "ur_PK", "en": "en_XX"
}
output_language_choices = ["English (en_XX)"] + [f"{name} ({code})" for code, name in translator.INDIAN_LANGUAGES.items()]
navigator_topic_choices = list(navigator.LEGAL_GUIDES.keys())
drafting_template_choices = list(drafting.DRAFTING_TEMPLATES.keys())
MAX_CLAUSES_TO_DISPLAY = 10

# --- 3. ENHANCED PROCESSING FUNCTIONS ---

def process_file_with_quality(file, progress=gr.Progress(track_tqdm=True)) -> Tuple[str, str, str, List]:
    """
    Enhanced file processing with quality awareness and proper state management.
    Returns: (preview_text, status_message, state_json, clause_updates)
    """
    if not file:
        app_state.document.reset()
        return _create_empty_response("Please upload a file first.")
    
    try:
        progress(0.1, desc="Extracting text content...")
        
        # Enhanced content extraction with quality assessment
        if hasattr(utils, 'enhanced_load_file_content'):
            extracted_text, quality_scores = utils.enhanced_load_file_content(file.name)
        else:
            # Fallback to original function
            extracted_text = utils.load_file_content(file.name)
            quality_scores = utils.assess_text_quality(extracted_text) if hasattr(utils, 'assess_text_quality') else {"overall": 0.5}
        
        if "Error:" in extracted_text:
            app_state.document.reset()
            app_state.document.error_message = extracted_text
            return _create_empty_response(extracted_text)
        
        # Update document state
        app_state.document.raw_text = extracted_text
        app_state.document.processed_text = extracted_text
        app_state.document.file_name = os.path.basename(file.name)
        app_state.document.file_path = file.name
        app_state.document.quality_score = quality_scores.get("overall", 0.5)
        app_state.document.quality_level = _get_quality_level(app_state.document.quality_score)
        app_state.document.processing_status = "Text extracted"
        app_state.document.error_message = ""
        
        progress(0.4, desc="Building knowledge base...")
        
        # Create knowledge base with quality awareness
        app_state.rag_agent.create_knowledge_base(
            extracted_text, 
            app_state.document.quality_score
        )
        app_state.document.has_knowledge_base = True
        app_state.document.processing_status = "Knowledge base created"
        
        progress(0.7, desc="Extracting legal clauses...")
        
        # Extract clauses
        try:
            found_clauses = clauses.extract_clauses(extracted_text, app_state.rag_agent)
            app_state.document.extracted_clauses = found_clauses
        except Exception as e:
            # Continue even if clause extraction fails
            app_state.document.extracted_clauses = []
            print(f"Warning: Clause extraction failed: {e}")
        
        progress(1.0, desc="Processing complete!")
        
        # Generate response
        preview_text = extracted_text[:3000] + ("..." if len(extracted_text) > 3000 else "")
        status_message = app_state.document.get_status_message()
        state_json = json.dumps(app_state.document.to_dict())
        clause_updates = _generate_clause_updates(app_state.document.extracted_clauses)
        
        return preview_text, status_message, state_json, clause_updates
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        app_state.document.reset()
        app_state.document.error_message = error_msg
        return _create_empty_response(error_msg)

def _get_quality_level(score: float) -> str:
    """Convert quality score to human-readable level."""
    if score >= 0.7:
        return "High"
    elif score >= 0.4:
        return "Medium" 
    else:
        return "Low"

def _create_empty_response(error_msg: str) -> Tuple[str, str, str, List]:
    """Create empty response for error cases."""
    clause_updates = []
    for _ in range(MAX_CLAUSES_TO_DISPLAY):
        clause_updates.extend([
            gr.update(visible=False),  # accordion
            gr.update(visible=False),  # content
            gr.update(visible=False)   # button
        ])
    return "", error_msg, json.dumps({}), clause_updates

def _generate_clause_updates(clauses: List[Dict]) -> List:
    """Generate Gradio updates for clause display."""
    updates = []
    for i in range(MAX_CLAUSES_TO_DISPLAY):
        if i < len(clauses):
            clause = clauses[i]
            risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(clause.get('risk', 'Unknown'), "⚪")
            updates.extend([
                gr.update(
                    label=f"{risk_color} {clause.get('title', f'Clause {i+1}')} (Risk: {clause.get('risk', 'Unknown')})", 
                    visible=True
                ),
                gr.update(
                    value=f"**Extracted Text:**\n\n> {clause.get('text', 'No text available')}", 
                    visible=True
                ),
                gr.update(visible=True)
            ])
        else:
            updates.extend([
                gr.update(visible=False),
                gr.update(visible=False), 
                gr.update(visible=False)
            ])
    return updates

def get_summary_with_state(state_json: str, progress=gr.Progress(track_tqdm=True)) -> str:
    """Enhanced summary generation with state awareness."""
    try:
        if not state_json:
            return "Please upload and process a document first."
        
        state_dict = json.loads(state_json)
        if not state_dict.get('processed_text') or not state_dict.get('has_knowledge_base'):
            return "Please upload and process a document first."
        
        progress(0, desc="Generating summary...")
        
        # Generate summary using the RAG agent
        summary = app_state.rag_agent.summarize()
        
        # Add quality context if relevant
        quality_score = state_dict.get('quality_score', 0.5)
        if quality_score < 0.5:
            quality_note = f"\n\n*Note: This summary is based on text with {quality_score:.1%} quality score due to OCR processing. Some details may be approximate.*"
            summary += quality_note
            
        return summary
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def respond_with_state(message: str, history: List, state_json: str, output_language: str) -> Tuple[str, List]:
    """Enhanced chat response with proper state management."""
    try:
        if not state_json:
            history.append((message, "Please upload and process a document first."))
            return "", history
        
        state_dict = json.loads(state_json)
        if not state_dict.get('has_knowledge_base'):
            history.append((message, "Please wait for document processing to complete."))
            return "", history
        
        # Language detection and translation
        target_lang_code = output_language.split('(')[-1].replace(')', '').strip()
        try:
            source_lang_code = LANG_CODE_MAPPING.get(detect(message), "en_XX")
        except:
            source_lang_code = "en_XX"
        
        # Translate query if needed
        english_query = (translator.translate_text(message, src_lang=source_lang_code, target_lang="en_XX") 
                        if source_lang_code != "en_XX" else message)
        
        # Get AI response
        english_answer = app_state.rag_agent.ask(english_query)
        
        # Add quality context for low-quality documents
        quality_score = state_dict.get('quality_score', 1.0)
        if quality_score < 0.5:
            confidence_note = f" *[Based on {quality_score:.0%} quality OCR text]*"
            english_answer += confidence_note
        
        # Translate response if needed
        final_answer = (translator.translate_text(english_answer, src_lang="en_XX", target_lang=target_lang_code)
                       if target_lang_code != "en_XX" else english_answer)
        
        history.append((message, final_answer))
        return "", history
        
    except Exception as e:
        error_response = f"An error occurred: {str(e)}"
        history.append((message, error_response))
        return "", history

def explain_clause_with_context(clause_text: str, state_json: str) -> gr.update:
    """Enhanced clause explanation with quality context."""
    try:
        if not clause_text or ">" not in clause_text:
            return gr.update(value="Could not find text to explain.", visible=True)
        
        core_text = clause_text.split(">")[-1].strip()
        
        # Create context-aware prompt
        quality_context = ""
        if state_json:
            try:
                state_dict = json.loads(state_json)
                quality_score = state_dict.get('quality_score', 1.0)
                if quality_score < 0.5:
                    quality_context = "\n\nNote: This text was extracted via OCR and may contain some recognition errors. Please focus on the general meaning."
            except:
                pass
        
        prompt = f"Please explain the following legal clause in simple, easy-to-understand terms:\n\n---\n{core_text}\n---{quality_context}"
        explanation = app_state.rag_agent.ask(prompt)
        
        return gr.update(value=explanation, visible=True)
        
    except Exception as e:
        return gr.update(value=f"Error explaining clause: {str(e)}", visible=True)

# Navigator and drafting functions (unchanged but with better error handling)
def load_guide_enhanced(topic: str) -> Tuple[str, List]:
    """Enhanced guide loading with error handling."""
    try:
        guide_text = navigator.LEGAL_GUIDES.get(topic)
        if guide_text:
            app_state.rag_agent.create_knowledge_base(guide_text)
            return f"✅ Guide for '{topic}' loaded successfully.", []
        return f"❌ Failed to load guide for '{topic}'.", []
    except Exception as e:
        return f"❌ Error loading guide: {str(e)}", []

def navigator_respond_enhanced(message: str, history: List) -> Tuple[str, List]:
    """Enhanced navigator response with error handling."""
    try:
        response = app_state.rag_agent.ask(message)
        history.append((message, response))
        return "", history
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        history.append((message, error_msg))
        return "", history

def generate_legal_draft_enhanced(template_name: str, *args) -> str:
    """Enhanced legal draft generation with error handling."""
    try:
        fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {})
        details = {label: arg for label, arg in zip(fields.keys(), args)}
        return drafting.generate_draft(template_name, details)
    except Exception as e:
        return f"❌ Error generating draft: {str(e)}"

# --- 4. GRADIO UI LAYOUT ---
with gr.Blocks(theme=gr.themes.Soft(), title="LegiWise AI - Enhanced") as demo:
    # Centralized state management
    doc_state_json = gr.State(value=json.dumps({}))
    
    gr.Markdown("# ⚖️ LegiWise AI – Your Enhanced Legal Assistant")
    gr.Markdown("*Now with advanced OCR quality awareness and improved state management*")
    
    with gr.Tabs():
        with gr.TabItem("📄 Document Analysis & Q&A"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="📎 Upload Legal Document")
                    status_box = gr.Textbox(
                        label="📊 Processing Status", 
                        interactive=False,
                        value="📄 No document uploaded"
                    )
                    summary_button = gr.Button("📋 Generate Summary", variant="secondary")
                    summary_output = gr.Textbox(label="📝 Document Summary", lines=10, interactive=False)
                    
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="💬 Document Chat", height=500)
                    with gr.Row():
                        output_language_dropdown = gr.Dropdown(
                            choices=output_language_choices, 
                            value="English (en_XX)", 
                            label="🌐 Answer Language", 
                            min_width=150
                        )
                        msg_box = gr.Textbox(
                            label="Ask a question...", 
                            show_label=False, 
                            container=False, 
                            scale=3,
                            placeholder="Ask me anything about the document..."
                        )

        with gr.TabItem("🧭 Legal Navigator"):
            with gr.Row():
                with gr.Column(scale=1):
                    navigator_dropdown = gr.Dropdown(choices=navigator_topic_choices, label="📚 Select Legal Topic")
                    navigator_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=2):
                    navigator_chatbot = gr.Chatbot(height=500, label="🗺️ Navigator Chat")
                    navigator_msg_box = gr.Textbox(
                        show_label=False, 
                        container=False,
                        placeholder="Ask about legal procedures and requirements..."
                    )
        
        with gr.TabItem("✍️ Legal Drafting Assistant"):
            with gr.Row():
                with gr.Column(scale=1):
                    drafting_dropdown = gr.Dropdown(choices=drafting_template_choices, label="📋 Document Type")
                    draft_button = gr.Button("✨ Generate Draft", variant="primary")
                    drafting_inputs = {}
                    unique_fields = {field for t in drafting.DRAFTING_TEMPLATES.values() for field in t["fields"].keys()}
                    for field in unique_fields:
                        drafting_inputs[field] = gr.Textbox(label=field, visible=False)
                with gr.Column(scale=2):
                    draft_output = gr.Textbox(label="📄 Generated Draft", lines=25, interactive=True)

        with gr.TabItem("🔍 Extracted Text & Clauses"):
            with gr.Row():
                text_preview_output = gr.Textbox(
                    label="📖 Extracted Text Preview", 
                    lines=25, 
                    interactive=False
                )
                with gr.Column():
                    gr.Markdown("### 🔍 Clause Analysis")
                    clause_outputs = []
                    for i in range(MAX_CLAUSES_TO_DISPLAY):
                        with gr.Accordion(f"Clause #{i+1}", visible=False) as acc:
                            content = gr.Markdown(visible=False)
                            explain_button = gr.Button("🤖 Explain with AI", visible=False)
                            explanation_box = gr.Textbox(label="💡 AI Explanation", interactive=False, visible=False)
                            # Enhanced clause explanation with state context
                            explain_button.click(
                                explain_clause_with_context, 
                                inputs=[content, doc_state_json], 
                                outputs=[explanation_box]
                            )
                            clause_outputs.extend([acc, content, explain_button])

    # --- 5. EVENT LISTENERS WITH ENHANCED STATE MANAGEMENT ---
    
    # File processing with quality awareness
    def process_and_update_ui(file):
        """Wrapper to handle the multiple outputs from file processing."""
        preview, status, state_json, clause_updates = process_file_with_quality(file)
        # Return all updates including the state
        return [preview, status, state_json] + clause_updates
    
    file_input.upload(
        process_and_update_ui,
        inputs=[file_input], 
        outputs=[text_preview_output, status_box, doc_state_json] + clause_outputs
    )
    
    # Enhanced event listeners
    summary_button.click(
        get_summary_with_state, 
        inputs=[doc_state_json], 
        outputs=[summary_output]
    )
    
    msg_box.submit(
        respond_with_state, 
        inputs=[msg_box, chatbot, doc_state_json, output_language_dropdown], 
        outputs=[msg_box, chatbot]
    )
    
    # Navigator events
    navigator_dropdown.change(
        load_guide_enhanced, 
        inputs=[navigator_dropdown], 
        outputs=[navigator_status, navigator_chatbot]
    )
    navigator_msg_box.submit(
        navigator_respond_enhanced, 
        inputs=[navigator_msg_box, navigator_chatbot], 
        outputs=[navigator_msg_box, navigator_chatbot]
    )
    
    # Drafting events
    def show_relevant_fields(template_name):
        template_fields = drafting.DRAFTING_TEMPLATES.get(template_name, {}).get("fields", {}).keys()
        updates = {}
        for field, component in drafting_inputs.items():
            updates[component] = gr.update(visible=field in template_fields)
        return updates
    
    all_field_components = list(drafting_inputs.values())
    drafting_dropdown.change(show_relevant_fields, inputs=[drafting_dropdown], outputs=all_field_components)
    draft_button.click(
        generate_legal_draft_enhanced, 
        inputs=[drafting_dropdown] + all_field_components, 
        outputs=[draft_output]
    )

# --- 6. LAUNCH ---
if __name__ == "__main__":
    if app_state.rag_agent is None:
        print(f"FATAL ERROR: Could not initialize RAG agent. Error: {app_state.initialization_error}")
        print("Please check your model paths and dependencies.")
    else:
        print("🚀 LegiWise AI Enhanced - Starting with improved state management...")
        demo.launch(share=True, server_name="0.0.0.0")