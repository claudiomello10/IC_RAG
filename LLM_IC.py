import pandas as pd
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import ujson as json
from transformers import TextIteratorStreamer
from threading import Thread


class LLM_IC:
    def __init__(
        self,
        embeddings_path: str = None,
        device: str = None,
        embeddings_model: str = "BAAI/bge-large-en",
        max_new_tokens: int = 2000,
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
                    "Book": item["Book"],
                    "Embedding": np.array(item["Embedding"]),
                }
            )

        # Create a dataframe
        self.embedding_df = pd.DataFrame(embedding_list)

        self.embeddings_model = SentenceTransformer("BAAI/bge-large-en", device=device)

        embeddings_dimension = self.embedding_df["Embedding"][0].shape[0]

        # Create a faiss index
        self.faiss_index = faiss.IndexFlatL2(embeddings_dimension)

        # Add the embeddings to the index
        embeddings = np.array(self.embedding_df["Embedding"].to_list())

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self.faiss_index.add(embeddings)

        # Load the model
        print("Loading the model")
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct",
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=False,
        )
        print("Model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/Phi-3-mini-128k-instruct"
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

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
        rag_text = "This is a context for the user query retrieved from several books and documents\n\n"
        retrieval_count = 1
        for index, row in search_results.iterrows():
            rag_text += f"Retriaval {retrieval_count}: From Book {row["Book"]} - Chapter {row['Chapter']} - {row['Title']}\n{row['Text']}\n\n"
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
        query = f"{query}\n\n if relevant cite the books chapters and sections that were used in the response"
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        output = self.pipe(messages, **self.generation_args)
        return output[0]["generated_text"]

    def generate_text_cite_stream_thread(self, query: str):
        rag_context = self.generate_rag_text(query)
        query = f"{query}\n\n if relevant cite the books from the context"
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        generation_kwargs = dict(
            text_inputs=messages,
            streamer=streamer,
            max_new_tokens=1500,
            temperature=0.0,
            do_sample=False,
        )

        return Thread(target=self.pipe, kwargs=generation_kwargs), streamer
