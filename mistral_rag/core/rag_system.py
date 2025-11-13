"""
Retrieval Augmented Generation (RAG) System

This module implements a RAG system using:
- Mistral-7B for language generation
- FAISS for vector storage
- LangChain for orchestration
- Web scraping for knowledge base
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
import nest_asyncio


class MistralRAGSystem:
    """
    RAG System using Mistral-7B

    Features:
    - Web scraping for knowledge base
    - Vector similarity search
    - Context-aware answering
    """

    def __init__(self, model_name='mistralai/Mistral-7B-Instruct-v0.1'):
        """
        Initialize RAG system

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

        # Allow nested async (needed for Colab)
        nest_asyncio.apply()

    def load_model(self, use_4bit=True):
        """
        Load Mistral model with quantization

        Args:
            use_4bit: Use 4-bit quantization for memory efficiency
        """
        print(f"Loading {self.model_name}...")

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        if use_4bit:
            # 4-bit quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        # Create text generation pipeline
        self.pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=0.2,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=1000,
        )

        # Wrap in LangChain HuggingFacePipeline
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)

        print("✓ Model loaded successfully!")

    def index_documents(self, urls, chunk_size=100, chunk_overlap=0):
        """
        Scrape and index documents from URLs

        Args:
            urls: List of URLs to scrape
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        print(f"Scraping {len(urls)} URLs...")

        # Load documents from URLs
        loader = AsyncChromiumLoader(urls)
        docs = loader.load()

        # Convert HTML to plain text
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunked_documents = text_splitter.split_documents(docs_transformed)

        print(f"Created {len(chunked_documents)} document chunks")

        # Create vector store
        print("Creating vector embeddings...")
        self.vectorstore = FAISS.from_documents(
            chunked_documents,
            HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        )

        # Create retriever
        self.retriever = self.vectorstore.as_retriever()

        print("✓ Documents indexed successfully!")

    def setup_rag_chain(self):
        """
        Setup the RAG chain for question answering
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.retriever is None:
            raise RuntimeError("No documents indexed. Call index_documents() first.")

        # Define prompt template
        prompt_template = """
### [INST] Instruction: Answer the question based on the provided context.
If you cannot answer based on the context, say so.

{context}

### QUESTION:
{question} [/INST]
"""

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        # Create LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # Create RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | llm_chain
        )

        print("✓ RAG chain configured!")

    def ask(self, question):
        """
        Ask a question using RAG

        Args:
            question: Question string

        Returns:
            Dictionary with answer and context
        """
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not setup. Call setup_rag_chain() first.")

        result = self.rag_chain.invoke(question)

        return {
            'question': question,
            'answer': result['text'],
            'context': result.get('context', [])
        }

    def ask_without_context(self, question):
        """
        Ask a question without RAG (direct LLM query)

        Args:
            question: Question string

        Returns:
            Answer string
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        prompt = f"### Question: {question}\n### Answer:"
        result = self.llm(prompt)

        return result
