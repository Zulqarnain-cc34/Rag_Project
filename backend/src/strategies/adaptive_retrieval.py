#!/usr/bin/env python
"""
adaptive_rag.py

This module implements an Adaptive Retrieval-Augmented Generation (RAG) system using various
retrieval strategies tailored to the query type (Factual, Analytical, Opinion, or Contextual).
It leverages langchain components, PostgreSQL with pgvector, and OpenAI's chat model for processing.
"""

import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any, Annotated
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from processor.file_processor import FileProcessor
from langchain_core.documents import Document  # Note: Duplicate import, ensure correct usage
from langchain_postgres import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_postgres.vectorstores import PGVector
from .agentic_chunker import AgenticChunker

# Load environment variables from a .env file
load_dotenv()

# PostgreSQL connection string with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
# Name of the collection to store documents
collection_name = "my_docs"

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Define required data models using Pydantic
class CategoriesOptions(BaseModel):
    """
    Represents the output of the query classification.
    Options: Factual, Analytical, Opinion, or Contextual.
    """
    category: str = Field(
        description="The category of the query. Options: Factual, Analytical, Opinion, or Contextual",
        example="Factual"
    )


class RelevantScore(BaseModel):
    """
    Represents the relevance score for a document with respect to a query.
    """
    score: float = Field(
        description="The relevance score of the document to the query",
        example=8.0
    )


class SelectedIndices(BaseModel):
    """
    Represents the indices of the selected documents after processing.
    """
    indices: List[int] = Field(
        description="Indices of selected documents",
        example=[0, 1, 2, 3]
    )


class SubQueries(BaseModel):
    """
    Represents a list of generated sub-queries for comprehensive analysis.
    """
    sub_queries: List[str] = Field(
        description="List of sub-queries for comprehensive analysis",
        example=["What is the population of New York?", "What is the GDP of New York?"]
    )


class QueryClassifier:
    """
    Classifies a query into one of four categories:
    Factual, Analytical, Opinion, or Contextual.
    """

    def __init__(self):
        # Initialize the ChatOpenAI model with the specified parameters.
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        # Prompt template for classifying the query
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="Classify the following query into one of these categories: Factual, Analytical, Opinion, or Contextual.\n"
                     "Query: {query}\nCategory:"
        )
        # Chain the prompt with the LLM to get a structured output.
        self.chain = self.prompt | self.llm.with_structured_output(CategoriesOptions)

    def classify(self, query: str) -> str:
        """
        Classify the input query and return its category.
        :param query: The query to classify.
        :return: Category of the query.
        """
        print("Classifying query...")
        return self.chain.invoke(query).category


class AgenticChunkerTextSplitter:
    def __init__(self) -> None:
        # Initialize the instance. No attributes are set in this simple constructor.
        pass

    def agentic_split_text(self, texts: list[str]) -> list[Annotated[Document, "external"]]:
        """
        Split a list of text strings into semantically meaningful chunks.

        This method iterates over each input text, uses an AgenticChunker to split the text into
        sentences (propositions), adds those propositions to the chunker, retrieves chunks from the
        chunker, and then wraps each chunk into a Document object with metadata.

        Args:
            texts (list[str]): A list of text strings to be processed.

        Returns:
            list[Annotated[Document, "external"]]: A list of Document objects containing the chunked text.
        """
        print("#### Agentic Text Splitting ####")
        # This will hold all the Document objects generated from the input texts.
        all_documents = []
        
        # Process each text in the provided list
        for text in texts:
            # Instantiate an AgenticChunker to perform the text chunking
            ac = AgenticChunker()
            # Split the text into sentences (propositions) for further processing.
            propositions = self._split_text_into_sentences(text)
            # Add the propositions to the AgenticChunker instance
            ac.add_propositions(propositions)
            # Retrieve the chunks as a list of strings
            chunks = ac.get_chunks(get_type="list_of_strings")
            # If chunks are not returned as a list, log a warning and set chunks to an empty list.
            if not isinstance(chunks, list):
                logging.warning("No chunks found for one of the texts.")
                chunks = []
            # Create a list of Document objects, ensuring each chunk is a string and 
            # adding a 'source' metadata field with the value "local".
            documents: list[Document] = [
                Document(page_content=str(chunk), metadata={"source": "local"}) for chunk in chunks
            ]
            # Extend the overall document list with the documents from the current text.
            all_documents.extend(documents)
        
        # Return the list of Document objects created from all texts.
        return all_documents

    def _split_text_into_sentences(self, text: str) -> list[str]:

        """
        Split the provided text into sentences.

        This helper method splits the input text on the period character ('.'),
        trims each resulting sentence, filters out any empty strings, and then
        appends a period back to each sentence.

        Args:
            text (str): The input text string to split.

        Returns:
            list[str]: A list of sentences with trailing periods.
        """
        # Split the text on periods, remove any empty strings, and reappend the period to each sentence.
        return [f"{sentence.strip()}." for sentence in text.split(".") if sentence.strip()]


class BaseRetrievalStrategy:
    """
    Base class for different retrieval strategies.
    It splits the input texts into documents, stores them in a PostgreSQL vector store,
    and provides a method to perform similarity search.
    """

    def __init__(self, texts: List[str]):
        # Initialize OpenAI embeddings.
        self.embeddings = OpenAIEmbeddings()
        # Split texts into documents with a specific chunk size and overlap.
        # text_splitter = SemanticChunker(
        #     OpenAIEmbeddings(), breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
        # )
        # self.documents = text_splitter.create_documents(texts)

        # text_splitter = text_splitter(
        # self.documents = text_splitter.create_documents(texts)
        text_splitter = AgenticChunkerTextSplitter()
        self.documents = text_splitter.agentic_split_text(texts)

        print(self.documents)
        # Add a dummy metadata field (an id) to each document.
        for i, doc in enumerate(self.documents, start=1):
            doc.metadata["id"] = i

        # Initialize the PostgreSQL vector store (PGVector) for storing document embeddings.
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )

        # Add documents to the vector store with their respective IDs.
        self.vector_store.add_documents(self.documents, ids=[doc.metadata["id"] for doc in self.documents])
        # Initialize the ChatOpenAI model.
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve documents similar to the query using similarity search.
        :param query: The query string.
        :param k: Number of documents to return.
        :return: A list of Document objects.
        """
        return self.vector_store.similarity_search(query, k=k)


class FactualRetrievalStrategy(BaseRetrievalStrategy):
    """
    Implements a retrieval strategy for factual queries.
    Enhances the query, performs a similarity search, and then ranks the documents.
    """

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Enhance the factual query, search for similar documents, rank them based on relevance,
        and return the top-k documents.
        :param query: The factual query.
        :param k: Number of documents to return.
        :return: A list of relevant Document objects.
        """
        print("Retrieving factual information...")
        # Enhance the query for better retrieval.
        enhanced_query_prompt = PromptTemplate(
            input_variables=["query"],
            template="Enhance this factual query for better information retrieval: {query}"
        )
        query_chain = enhanced_query_prompt | self.llm
        enhanced_query = query_chain.invoke(query).content
        print(f'Enhanced query: {enhanced_query}')

        # Retrieve more documents than needed initially.
        docs = self.vector_store.similarity_search(enhanced_query, k=k * 2)

        # Create a prompt to rank the relevance of each document.
        ranking_prompt = PromptTemplate(
            input_variables=["query", "doc"],
            template="On a scale of 1-10, how relevant is this document to the query: '{query}'?\n"
                     "Document: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        print("Ranking documents...")
        # Rank each document based on the response from the LLM.
        for doc in docs:
            input_data = {"query": enhanced_query, "doc": doc.page_content}
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort the documents based on the relevance score.
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class AnalyticalRetrievalStrategy(BaseRetrievalStrategy):
    """
    Implements a retrieval strategy for analytical queries.
    Generates sub-queries for the main query, performs searches, and then selects a diverse set of documents.
    """

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Generate sub-queries from the main query, retrieve documents for each sub-query, and then select
        a diverse set of top-k documents.
        :param query: The analytical query.
        :param k: Number of documents to return.
        :return: A list of relevant Document objects.
        """
        print("Retrieving analytical information...")
        # Generate sub-questions for the analytical query.
        sub_queries_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Generate {k} sub-questions for: {query}"
        )
        sub_queries_chain = sub_queries_prompt | self.llm.with_structured_output(SubQueries)
        input_data = {"query": query, "k": k}
        sub_queries = sub_queries_chain.invoke(input_data).sub_queries
        print(f'Sub-queries: {sub_queries}')

        all_docs = []
        # Retrieve documents for each sub-query.
        for sub_query in sub_queries:
            all_docs.extend(self.vector_store.similarity_search(sub_query, k=2))

        # Select a diverse set of documents from the pool.
        diversity_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="Select the most diverse and relevant set of {k} documents for the query: '{query}'\n"
                     "Documents: {docs}\n"
        )
        diversity_chain = diversity_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:50]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = diversity_chain.invoke(input_data).indices

        return [all_docs[i] for i in selected_indices if i < len(all_docs)]


class OpinionRetrievalStrategy(BaseRetrievalStrategy):
    """
    Implements a retrieval strategy for opinion-based queries.
    Identifies distinct viewpoints, retrieves documents for each viewpoint, and then selects the most representative ones.
    """

    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """
        Identify distinct viewpoints on the topic, perform searches for each viewpoint,
        and select the top-k diverse documents.
        :param query: The opinion-based query.
        :param k: Number of documents to return.
        :return: A list of relevant Document objects.
        """
        print("Retrieving opinions...")
        # Generate distinct viewpoints for the query.
        viewpoints_prompt = PromptTemplate(
            input_variables=["query", "k"],
            template="Identify {k} distinct viewpoints or perspectives on the topic: {query}"
        )
        viewpoints_chain = viewpoints_prompt | self.llm
        input_data = {"query": query, "k": k}
        viewpoints = viewpoints_chain.invoke(input_data).content.split('\n')
        print(f'Viewpoints: {viewpoints}')

        all_docs = []
        # Retrieve documents by combining the query with each viewpoint.
        for viewpoint in viewpoints:
            all_docs.extend(self.vector_store.similarity_search(f"{query} {viewpoint}", k=2))

        # Group and select representative documents based on the retrieved opinions.
        opinion_prompt = PromptTemplate(
            input_variables=["query", "docs", "k"],
            template="Classify these documents into distinct opinions on '{query}' and select the {k} most representative and diverse viewpoints:\n"
                     "Documents: {docs}\nSelected indices:"
        )
        opinion_chain = opinion_prompt | self.llm.with_structured_output(SelectedIndices)
        docs_text = "\n".join([f"{i}: {doc.page_content[:100]}..." for i, doc in enumerate(all_docs)])
        input_data = {"query": query, "docs": docs_text, "k": k}
        selected_indices = opinion_chain.invoke(input_data).indices

        # Return documents based on the indices, ensuring valid integer conversion.
        return [all_docs[int(i)] for i in selected_indices if isinstance(i, int) or (isinstance(i, str) and i.isdigit()) and int(i) < len(all_docs)]


class ContextualRetrievalStrategy(BaseRetrievalStrategy):
    """
    Implements a retrieval strategy for contextual queries.
    Reformulates the query based on user context, performs a similarity search,
    and then ranks the documents.
    """

    def retrieve(self, query: str, k: int = 4, user_context: str = None) -> List[Document]:
        """
        Reformulate the query with the user context, retrieve similar documents,
        rank them, and return the top-k documents.
        :param query: The contextual query.
        :param k: Number of documents to return.
        :param user_context: Additional user context to tailor the query.
        :return: A list of relevant Document objects.
        """
        print("Retrieving contextual information...")
        # Reformulate the query with user context.
        context_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the user context: {context}\nReformulate the query to best address the user's needs: {query}"
        )
        context_chain = context_prompt | self.llm
        input_data = {"query": query, "context": user_context or "No specific context provided"}
        contextualized_query = context_chain.invoke(input_data).content
        print(f'Contextualized query: {contextualized_query}')

        # Retrieve documents based on the contextualized query.
        docs = self.vector_store.similarity_search(contextualized_query, k=k * 2)

        # Create a prompt to rank the relevance of each document given the context.
        ranking_prompt = PromptTemplate(
            input_variables=["query", "context", "doc"],
            template="Given the query: '{query}' and user context: '{context}', rate the relevance of this document on a scale of 1-10:\n"
                     "Document: {doc}\nRelevance score:"
        )
        ranking_chain = ranking_prompt | self.llm.with_structured_output(RelevantScore)

        ranked_docs = []
        for doc in docs:
            input_data = {
                "query": contextualized_query,
                "context": user_context or "No specific context provided",
                "doc": doc.page_content
            }
            score = float(ranking_chain.invoke(input_data).score)
            ranked_docs.append((doc, score))

        # Sort documents based on the computed relevance score.
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:k]]


class AdaptiveRAG:
    """
    Adaptive Retrieval-Augmented Generation (RAG) system that:
    - Classifies the query type.
    - Uses the appropriate retrieval strategy.
    - Generates an answer using a language model with provided context.
    """

    def __init__(self, texts: List[str]):
        """
        Initialize the AdaptiveRAG system.
        :param texts: A list of texts to be processed and stored.
        """
        # Initialize the query classifier.
        self.classifier = QueryClassifier()
        # Map each query category to its retrieval strategy.
        self.strategies = {
            "Factual": FactualRetrievalStrategy(texts),
            "Analytical": AnalyticalRetrievalStrategy(texts),
            "Opinion": OpinionRetrievalStrategy(texts),
            "Contextual": ContextualRetrievalStrategy(texts)
        }
        # Initialize the ChatOpenAI model.
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
        # Define a prompt template for answer generation.
        prompt_template = (
            "Use the following pieces of context to answer the question at the end.\n\n"
            "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n"
            "{context}\n\nQuestion: {question}\nAnswer:"
        )
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm_chain = self.prompt | self.llm

    def answer(self, query: str) -> str:
        """
        Process the query by classifying it, retrieving relevant documents, and generating an answer.
        :param query: The input query.
        :return: The generated answer as a string.
        """
        # Classify the query into a category.
        category = self.classifier.classify(query)
        # For adaptive purposes, here the Contextual strategy is chosen by default.
        # You may update this to dynamically select a strategy based on the category.
        strategy = self.strategies[category]
        # Retrieve relevant documents.
        docs = strategy.retrieve(query)
        # Prepare the input for the language model using the retrieved context.
        input_data = {"context": "\n".join([doc.page_content for doc in docs]), "question": query}
        # Generate and return the answer.
        return self.llm_chain.invoke(input_data).content
