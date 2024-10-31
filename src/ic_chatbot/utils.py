import pandas as pd
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import ujson as json
from openai import OpenAI
import os


class LLM_IC:
    """
    LLM_IC class for handling language model interactions and retrieval-augmented generation (RAG).
    Attributes:
        device (torch.device): The device to run the model on (CPU or GPU).
        embeddings_path (str): Path to the embeddings file.
        model (str): The language model to use.
        messages (list): List to store messages.
        embedding_df (pd.DataFrame): DataFrame containing the embeddings and related information.
        embeddings_model (SentenceTransformer): The embeddings model.
        faiss_index (faiss.IndexFlatL2): The FAISS index for similarity search.
        client (OpenAI): The OpenAI client for generating responses.
    Methods:
        __init__(embeddings_path: str, device: str = None, embeddings_model: str = "BAAI/bge-large-en", model: str = "gpt-4o-mini"):
            Initializes the LLM_IC class with the given parameters and loads the embeddings and models.
        search(query: str, top_k: int = 5):
            Searches for the top_k most similar embeddings to the query and returns the corresponding DataFrame rows.
        generate_embeddings(query: str):
            Generates embeddings for the given query using the embeddings model.
        decode(embedding: np.array):
            Decodes the given embedding back to text using the embeddings model.
        generate_rag_text(query: str, top_k: int = 5):
            Generates a retrieval-augmented generation (RAG) text based on the query and top_k search results.
        generate_response(query: str, model: str | None = None):
            Generates a response to the query using the language model and RAG context.
        generate_response_stream(query: str, model: str | None = None):
            Generates a streaming response to the query using the language model and RAG context.
    """

    def __init__(
        self,
        embeddings_paths: list,
        device: str = None,
        embeddings_model: str = "jinaai/jina-embeddings-v3",
        model: str = "gpt-4o-mini",
    ):
        """
        Initializes the LLM_IC class with the given parameters and loads the embeddings and models.
        Args:
            embeddings_path (str): Path to the embeddings file.
            device (str): The device to run the model on (CPU or GPU).
            embeddings_model (str): The SentenceTransformer model to use for generating embeddings.
            model (str): The language model to use for generating responses.
        Returns:
            LLM_IC object.
        """

        # Raise error if the embeddings path does not exist
        embedding_list = []
        for embeddings_path in embeddings_paths:
            if not os.path.exists(embeddings_path):
                raise FileNotFoundError(
                    f"Embeddings file not found at {embeddings_path}"
                )
            with open(embeddings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Create a list of dictionaries

            for item in data:
                embedding_list.append(
                    {
                        "Chapter": item["Chapter"],
                        "Text": item["Text"],
                        "Embedding": np.array(item["Embedding"]),
                        "Topic": item["Topic"],
                        "Book": item["Book"],
                    }
                )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.embeddings_path = embeddings_path
        self.model = model
        self.messages = []

        # Load the json file
        try:

            # Create a dataframe
            self.embedding_df = pd.DataFrame(embedding_list)
        except Exception as e:
            print(f"Error loading the embeddings file: {e}")

        # Load the embeddings model
        try:
            print("Loading the embeddings model")
            self.embeddings_model = SentenceTransformer(
                embeddings_model, device=device, trust_remote_code=True
            )
            print("Embeddings model loaded")
        except Exception as e:
            print(f"Error loading the embeddings model: {e}")

        # Create a faiss index
        try:
            # Get the embeddings dimension
            embeddings_dimension = np.array(self.embedding_df["Embedding"][0]).shape[0]

            # Create a faiss index
            self.faiss_index = faiss.IndexFlatIP(embeddings_dimension)

            # Add the embeddings to the index
            embeddings = np.array(self.embedding_df["Embedding"].to_list())

            # Check if the embeddings are 1D
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            # Add the embeddings to the faiss index
            self.faiss_index.add(embeddings)

        except Exception as e:
            print(f"Error creating the faiss index: {e}")
        self.client = OpenAI()

    def search(self, query: str, top_k: int = 5):
        """
        Searches for the top_k most similar embeddings to the query and returns the corresponding DataFrame rows.
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
        Returns:
            pd.DataFrame: The DataFrame containing the top_k search results.
        """
        query_embedding = self.generate_embeddings(query)
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        return self.embedding_df.iloc[indices[0]]

    def generate_embeddings(self, query: str):
        """
        Generates embeddings for the given query using the embeddings model.
        Args:
            query (str): The query to generate embeddings for.
        Returns:
            np.array: The embeddings for the query.
        """

        embeddings = self.embeddings_model.encode(query)
        embeddings = np.array(embeddings)
        return embeddings

    def decode(self, embedding: np.array):
        """ "
        Decodes the given embedding back to text using the embeddings model.
        Args:
            embedding (np.array): The embedding to decode.
        Returns:
            str: The decoded text from the embedding.
        """
        return self.embeddings_model.decode(embedding)

    def generate_rag_text(self, query: str, top_k: int = 5):
        """
        Generates a retrieval-augmented generation (RAG) text based on the query and top_k search results.
        Args:
            query (str): The query to generate RAG text for.
            top_k (int): The number of top results to use in the RAG text.
        Returns:
            str: The RAG text generated based on the query and search results.
        """
        search_results = self.search(query, top_k)
        rag_text = """You are an assistant helping a student to study machine learning.
        The student asks you a question and you provide an answer and a citation to the books from the retrieval-augmented generation (RAG) context, give the names chapters and sections of the books that should help him.
        When giving him the name of the book, you should provide the full name of the book.
        When giving him the name of the chapter, you should provide the full name of the chapter.
        When giving him the name of the section, you should provide the full name of the section.
        The citation should include the chapter and section of the book that was used to generate the answer.
        If the question is about a specific topic in the books, cite the chapter and section that defines the topic.
        If the student asks you a question that requires mathematical calculations do not provide the answer, provide only the method to solve the problem step by step, and instruct him where to find the solution in the book.
        Here is the context for the user query retrieved from the books:

        """
        retrieval_count = 1
        for index, row in search_results.iterrows():
            rag_text += f"Retriaval {retrieval_count}: From Book {row['Book']} - From Chapter {row['Chapter']} - Section: {row['Topic']}\n{row['Text']}\n\n"
            retrieval_count += 1
        return rag_text

    def generate_response(self, query: str, model: str | None = None):
        """
        Generates a response to the query using the language model and RAG context.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            str: The response generated by the language model.
        """
        if model is None:
            model = self.model
        rag_context = self.generate_rag_text(query)
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        # Get OPENAI_API_KEY from the environment
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def generate_response_conversation(self, query: str, model: str | None = None):
        """
        Generates a response to the query using the language model, RAG context, and conversation history.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            str: The response generated by the language model.
        """
        if model is None:
            model = self.model
        rag_context = self.generate_rag_text(query)
        self.messages.append({"role": "system", "content": rag_context})
        self.messages.append({"role": "user", "content": query})
        messages = self.messages
        # Get OPENAI_API_KEY from the environment
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        self.messages.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )
        return completion.choices[0].message.content

    def generate_response_stream(self, query: str, model: str | None = None):
        """
        Generates a streaming response to the query using the language model and RAG context.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            response: The streaming response generated by the language model.
        """
        if model is None:
            model = self.model
        rag_context = self.generate_rag_text(query)
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        # Get OPENAI_API_KEY from the environment
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        return response
