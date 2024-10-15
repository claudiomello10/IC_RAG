import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from LLM_IC import LLM_IC


# Remove the warnings
import warnings

warnings.filterwarnings("ignore")


# Create the LLM_IC object
llm = LLM_IC(embeddings_path="full_df_embeddings.json")

# Get the input
query = input("LLM_IC: ")

# Generate the text
print(llm.generate_text_cite(query))
# print(llm.generate_text(query))
