# importing required modules
from pypdf import PdfReader
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from langchain.text_splitter import NLTKTextSplitter
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()


tqdm.pandas()

# Set the device to 'cuda' if you have a GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


df = pd.read_csv("all_chunks.csv")


# Create an embedding space for the chunks using
embedding_model = SentenceTransformer(
    "jinaai/jina-embeddings-v3", device=device, trust_remote_code=True
)


# Create the embeddings for the chunks
print("Creating embeddings for the chunks")
df["Embedding"] = df["Text"].progress_apply(lambda x: embedding_model.encode(x))

print("Saving the embeddings to the files")

# Dvidide the dataframe into two parts
df1 = df.iloc[: len(df) // 2]
df2 = df.iloc[len(df) // 2 :]
df1.to_json("full_df_embeddings_sections1.json", orient="records")
df2.to_json("full_df_embeddings_sections2.json", orient="records")
