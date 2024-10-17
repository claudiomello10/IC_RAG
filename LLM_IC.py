import pandas as pd
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ujson as json
from openai import OpenAI
import os


class LLM_IC:
    def __init__(
        self,
        embeddings_path: str = None,
        device: str = None,
        embeddings_model: str = "BAAI/bge-large-en",
        max_new_tokens: int = 2000,
        load_model: bool = True,
    ):

        if embeddings_path is None:
            raise ValueError("embeddings_path is required")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.embeddings_path = embeddings_path

        # Load the json file
        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a list of dictionaries
        embedding_list = []
        for item in data:
            embedding_list.append(
                {
                    "Chapter": item["Chapter"],
                    "Title": item["Title"],
                    "Text": item["Text"],
                    "Embedding": np.array(item["Embedding"]),
                }
            )

        # Create a dataframe
        self.embedding_df = pd.DataFrame(embedding_list)

        # Load the embeddings model
        print("Loading the embeddings model")
        self.embeddings_model = SentenceTransformer("BAAI/bge-large-en", device=device)
        print("Embeddings model loaded")

        embeddings_dimension = self.embedding_df["Embedding"][0].shape[0]

        # Create a faiss index
        self.faiss_index = faiss.IndexFlatL2(embeddings_dimension)

        # Add the embeddings to the index
        embeddings = np.array(self.embedding_df["Embedding"].to_list())

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.faiss_index.add(embeddings)

        # Load the model
        if load_model:
            print("Loading the model")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Phi-3-mini-128k-instruct",
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=False,
            )
            print("Model loaded")

            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/Phi-3-mini-128k-instruct"
            )

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )

            self.generation_args = {
                "max_new_tokens": max_new_tokens,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
        else:
            self.pipe = None
            print("Model not loaded")

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.generate_embeddings(query)
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        return self.embedding_df.iloc[indices[0]]

    def generate_embeddings(self, query: str):
        return self.embeddings_model.encode(query)

    def decode(self, embedding: np.array):
        return self.embeddings_model.decode(embedding)

    def generate_rag_text(self, query: str, top_k: int = 5):
        search_results = self.search(query, top_k)
        rag_text = "This is a context for the user query retrieved from the Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow book\n\n"
        retrieval_count = 1
        for index, row in search_results.iterrows():
            rag_text += f"Retriaval {retrieval_count}: From Chapter {row['Chapter']} - {row['Title']}\n{row['Text']}\n\n"
            retrieval_count += 1
        return rag_text

    def generate_text(self, query: str):
        rag_context = self.generate_rag_text(query)
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        output = self.pipe(messages, **self.generation_args)
        return output[0]["generated_text"]

    def generate_text_cite(self, query: str):
        rag_context = self.generate_rag_text(query)
        query = f"{query}\n\n if relevant cite chapters and sections of the book that were used in the response"
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        output = self.pipe(messages, **self.generation_args)
        return output[0]["generated_text"]


class LLM_IC_OPENAI:
    def __init__(
        self,
        embeddings_path: str = None,
        device: str = None,
        embeddings_model: str = "BAAI/bge-large-en",
        model: str = "gpt-4o-mini",
    ):

        if embeddings_path is None:
            raise ValueError("embeddings_path is required")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.embeddings_path = embeddings_path
        self.model = model

        # Load the json file
        with open(embeddings_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Create a list of dictionaries
        embedding_list = []
        for item in data:
            embedding_list.append(
                {
                    "Chapter": item["Chapter"],
                    "Title": item["Title"],
                    "Text": item["Text"],
                    "Embedding": np.array(item["Embedding"]),
                }
            )

        # Create a dataframe
        self.embedding_df = pd.DataFrame(embedding_list)

        # Load the embeddings model
        print("Loading the embeddings model")
        self.embeddings_model = SentenceTransformer(embeddings_model, device=device)
        print("Embeddings model loaded")

        embeddings_dimension = self.embedding_df["Embedding"][0].shape[0]

        # Create a faiss index
        self.faiss_index = faiss.IndexFlatL2(embeddings_dimension)

        # Add the embeddings to the index
        embeddings = np.array(self.embedding_df["Embedding"].to_list())

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.faiss_index.add(embeddings)
        self.client = OpenAI()

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.generate_embeddings(query)
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        return self.embedding_df.iloc[indices[0]]

    def generate_embeddings(self, query: str):
        return self.embeddings_model.encode(query)

    def decode(self, embedding: np.array):
        return self.embeddings_model.decode(embedding)

    def generate_rag_text(self, query: str, top_k: int = 5):
        search_results = self.search(query, top_k)
        rag_text = """You are an assistant helping a student to study machine learning.
        The student asks you a question and you provide an answer with a citation to the book "Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.
        The citation should include the chapter and section of the book that was used to generate the answer.
        If the student asks you a question that requires mathematical calculations provide only the method to solve the problem step by step, and instruct him where to find the solution in the book.
        Here is the context for the user query retrieved from the book:

        """
        retrieval_count = 1
        for index, row in search_results.iterrows():
            rag_text += f"Retriaval {retrieval_count}: From Chapter {row['Chapter']} - {row['Title']}\n{row['Text']}\n\n"
            retrieval_count += 1
        return rag_text

    def generate_text_cite(self, query: str, model: str | None = None):
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
