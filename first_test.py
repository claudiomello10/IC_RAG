import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from LLM_IC import LLM_IC
from transformers import TextIteratorStreamer
from threading import Thread


# Remove the warnings
import warnings

warnings.filterwarnings("ignore")


# Create the LLM_IC object
llm = LLM_IC(embeddings_path=r"Data\intermediate\full_df_embeddings.json")

# Get the input
query = input("\nUser: ")

# Generate the response
thread, streamer = llm.generate_text_cite_stream_thread(query)

print("\n\nLLM_IC: ", end="")
thread.start()
for new_text in streamer:
    print(new_text, end="")
