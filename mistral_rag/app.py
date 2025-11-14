"""
Mistral RAG System Web Interface

Interactive question-answering system with:
- Document indexing from URLs
- Context-aware responses
- Comparison with/without RAG
"""

import gradio as gr
from core.rag_system import MistralRAGSystem


# Global RAG system instance
rag_system = None


def initialize_system(progress=gr.Progress()):
    """Initialize the RAG system"""
    global rag_system

    try:
        progress(0, desc="Initializing...")
        rag_system = MistralRAGSystem()

        progress(0.3, desc="Loading model...")
        rag_system.load_model()

        progress(1.0, desc="Ready!")
        return "✓ System initialized successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def index_documents(urls_text, progress=gr.Progress()):
    """Index documents from URLs"""
    global rag_system

    if rag_system is None:
        return "❌ Please initialize the system first!"

    try:
        # Parse URLs
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]

        if not urls:
            return "❌ Please provide at least one URL"

        progress(0, desc="Scraping documents...")
        rag_system.index_documents(urls)

        progress(0.7, desc="Setting up RAG chain...")
        rag_system.setup_rag_chain()

        progress(1.0, desc="Complete!")
        return f"✓ Indexed {len(urls)} URL(s) successfully!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def ask_question_rag(question, progress=gr.Progress()):
    """Ask question with RAG"""
    global rag_system

    if rag_system is None:
        return "Please initialize system first!", ""

    try:
        progress(0.5, desc="Searching context...")
        result = rag_system.ask(question)

        # Format context (result['context'] is now a list of strings)
        context_text = "\n\n".join([
            f"**Source {i+1}:**\n{content[:200]}..."
            for i, content in enumerate(result['context'][:3])
        ])

        # Add source count
        context_text = f"*Found {result['num_sources']} relevant source(s)*\n\n" + context_text

        progress(1.0, desc="Complete!")
        return result['answer'], context_text
    except Exception as e:
        return f"Error: {str(e)}", ""


def ask_question_no_rag(question, progress=gr.Progress()):
    """Ask question without RAG"""
    global rag_system

    if rag_system is None:
        return "Please initialize system first!"

    try:
        progress(0.5, desc="Generating answer...")
        answer = rag_system.ask_without_context(question)
        progress(1.0, desc="Complete!")
        return answer
    except Exception as e:
        return f"Error: {str(e)}"


def create_interface():
    """Create Gradio interface"""

    with gr.Blocks(title="Mistral RAG", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🤖 Mistral RAG: Context-Aware Question Answering

        Ask questions and get answers based on your own documents!

        **How it works:**
        1. Initialize the Mistral-7B model
        2. Provide URLs to scrape and index
        3. Ask questions - the system retrieves relevant context and generates answers

        **RAG (Retrieval Augmented Generation)** combines:
        - 🔍 Vector similarity search to find relevant information
        - 🧠 Large language model to generate coherent answers
        """)

        with gr.Tab("🚀 Setup"):
            gr.Markdown("### Step 1: Initialize Model")

            init_btn = gr.Button("Initialize System", variant="primary", size="lg")
            init_status = gr.Textbox(label="Status", interactive=False)

            init_btn.click(initialize_system, outputs=init_status)

            gr.Markdown("### Step 2: Index Documents")

            urls_input = gr.Textbox(
                label="URLs (one per line)",
                placeholder="https://example.com/article1\nhttps://example.com/article2",
                lines=5,
                value="https://www.fantasypros.com/2023/11/rival-fantasy-nfl-week-10/"
            )

            index_btn = gr.Button("Index Documents", variant="primary")
            index_status = gr.Textbox(label="Indexing Status", interactive=False)

            index_btn.click(index_documents, inputs=urls_input, outputs=index_status)

        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Ask Questions with RAG")

            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the indexed documents...",
                lines=2
            )

            with gr.Row():
                ask_rag_btn = gr.Button("Ask with RAG", variant="primary")
                ask_no_rag_btn = gr.Button("Ask without RAG (Direct LLM)")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Answer")
                    answer_output = gr.Textbox(label="", lines=10, show_label=False)

                with gr.Column():
                    gr.Markdown("#### Retrieved Context")
                    context_output = gr.Markdown()

            ask_rag_btn.click(
                ask_question_rag,
                inputs=question_input,
                outputs=[answer_output, context_output]
            )

            ask_no_rag_btn.click(
                ask_question_no_rag,
                inputs=question_input,
                outputs=answer_output
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About RAG (Retrieval Augmented Generation)

            RAG combines the best of both worlds:

            ### 🔍 Retrieval
            - Scrapes and indexes web content
            - Splits documents into chunks
            - Creates vector embeddings
            - Finds most relevant chunks using similarity search

            ### 🧠 Generation
            - Uses retrieved context as grounding
            - Generates factual, relevant answers
            - Reduces hallucinations
            - Provides source attribution

            ### Architecture:
            ```
            Question → Vector Search → Relevant Docs → LLM + Context → Answer
            ```

            ### Components:
            - **Model**: Mistral-7B-Instruct (4-bit quantized)
            - **Embeddings**: sentence-transformers/all-mpnet-base-v2
            - **Vector Store**: FAISS
            - **Framework**: LangChain (modern LCEL patterns)

            ### Benefits:
            - ✅ Grounded in actual documents
            - ✅ Reduces hallucinations
            - ✅ Can cite sources
            - ✅ Updates knowledge without retraining
            - ✅ Domain-specific expertise

            ### Example Use Cases:
            - Customer support with product documentation
            - Research assistant for academic papers
            - Legal document analysis
            - Medical information retrieval
            - News article summarization
            """)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)
