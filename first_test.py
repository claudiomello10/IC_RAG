import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from LLM_IC import LLM_IC
from transformers import TextIteratorStreamer
from threading import Thread


# Remove the warnings
import warnings

warnings.filterwarnings("ignore")


# Create the LLM_IC object
llm = LLM_IC(embeddings_path="full_df_embeddings.json")

# Get the input
query = input("LLM_IC: ")

# Generate the text

rag_context = llm.generate_rag_text(query)
query = f"{query}\n\n if relevant cite the books chapters and sections that were used in the response"
messages = [
    {"role": "system", "content": rag_context},
    {"role": "user", "content": query},
]

inputs = tokenizer.apply_chat_template(messages)
inputs["input_ids"] = inputs["input_ids"]  # .to("cuda")


streamer = TextIteratorStreamer(llm.tokenizer, skip_prompt=True)

generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=1500)

thread = Thread(target=llm.model.generate, kwargs=generation_kwargs)

output = llm.pipe(messages, **llm.generation_args, stream=True)

# Print the output

print(output[0]["generated_text"])
