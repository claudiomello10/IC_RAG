from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import os


documents = SimpleDirectoryReader("data/Books").load_data(show_progress=True)


# Get the current working directory
cwd = os.getcwd()

# Define the cache folder
cache_folder = os.path.join(cwd, "cache")

# Create the cache folder if it doesn't exist
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)


index = VectorStoreIndex.from_documents(documents, show_progress=True)


index.storage_context.persist("./index/llama_index")
