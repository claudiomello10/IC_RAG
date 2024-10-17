import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from LLM_IC import LLM_IC_OPENAI


# Remove the warnings
import warnings

warnings.filterwarnings("ignore")


# Create the LLM_IC object
llm = LLM_IC_OPENAI(embeddings_path="full_df_embeddings.json")

while True:
    # Get the input
    query = input("LLM_IC: ")

    # Generate the text
    print(llm.generate_text_cite(query), end="\n\n")
    # print(llm.generate_rag_text(query))
