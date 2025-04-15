
# Import required libraries
import os  # For interacting with the operating system
import sys  # For system-specific parameters and functions
from dotenv import load_dotenv  # To load environment variables from a .env file
from typing import List, Dict, Any, Annotated  # Type hinting tools
from pydantic import BaseModel, Field  # For data validation and settings management

# Import necessary components from the phi framework
from phi.agent import Agent  # Main agent interface
from phi.embedder.openai import OpenAIEmbedder  # Embedding model from OpenAI
from phi.document.chunking.agentic import AgenticChunking  # Document chunking strategy
from phi.model.openai import OpenAIChat  # OpenAI chat model interface
from phi.knowledge.text import TextKnowledgeBase  # Text-based knowledge base
from phi.vectordb.pgvector import PgVector  # PostgreSQL vector store integration

# Load environment variables from a .env file in the project root
load_dotenv()

# Define PostgreSQL database connection string with pgvector enabled
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"

# Set the OpenAI API key using the value loaded from the .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


class BaseRetrievalStrategy:
    """
    Base class for retrieval strategies in an RAG (Retrieval-Augmented Generation) setup.
    This class:
    - Loads processed documents.
    - Stores and indexes them in a vector database.
    - Provides a method to query the documents using semantic similarity.
    """

    def __init__(self):
        # Create a text-based knowledge base with OpenAI embeddings and pgvector
        self.knowledge_base = TextKnowledgeBase(
            path="/home/alpha/program_files/Projects/project_ongoing/project_Umbradge_Slackbot/backend/src/data/processed_docs/",  # Path to preprocessed documents
            vector_db=PgVector(
                table_name='recipe_s4',  # Table name for storing vector data
                db_url=connection,  # PostgreSQL connection string
                embedder=OpenAIEmbedder(model="text-embedding-3-small")  # OpenAI model for embedding text
            ),
            chunking_strategy=AgenticChunking()  # Strategy to break large documents into manageable chunks
        )

        # Create an agent with a chat model and link it to the knowledge base
        self.agent = Agent(
            model=OpenAIChat(id="gpt-4o"),  # Use OpenAI GPT-4o model
            search_knowledge=True,  # Enable knowledge-based search
            knowledge_base=self.knowledge_base,  # Use the defined knowledge base
            show_tool_calls=True,  # Display tool calls (useful for debugging)
            markdown=True  # Enable markdown formatting in responses
        )

        # Load the knowledge base into memory (without recreating if already present)
        # self.agent.knowledge.load(recreate=False)

    def retrieve(self, query):
        """
        Perform a semantic similarity search using the agent and return the response.
        """
        self.agent.print_response(query, markdown=True)
        print(self.agent.memory.messages[-1].content)
        return self.agent.memory.messages[-1].content


class AdaptiveRAG:
    """
    Wrapper class for adaptive retrieval-augmented generation (RAG).
    Uses a base strategy to perform retrieval and generate answers.
    """

    def __init__(self):
        # Initialize retrieval strategy (can be extended to support multiple)
        self.strategies = BaseRetrievalStrategy()

    def answer(self, query: str) -> str:
        """
        Answer the query using the current retrieval strategy.
        """
        docs = self.strategies.retrieve(query)
        return docs  # Return the final response from the agent
