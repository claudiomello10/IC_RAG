import os
import inquirer

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from LLM_IC import LLM_IC_OPENAI


questions = [
    inquirer.List(
        "run_mode",
        message="Select response mode:",
        choices=[
            ("Stream the response", True),
            ("Get the full answer in one time", False),
        ],
    ),
]

# Remove the warnings
import warnings

warnings.filterwarnings("ignore")


# Create the LLM_IC object
llm = LLM_IC_OPENAI(embeddings_path="full_df_embeddings.json")

# Ask the user for the mode
answer = inquirer.prompt(questions)

if answer["run_mode"]:
    while True:
        # Get the input
        query = input("\nUser: ")

        response = llm.generate_response_stream(query)
        print("\n\nLLM_IC: ", end="")
        # Print the stream as it comes
        for r in response:
            if r.choices[0].delta.content != None:
                print(r.choices[0].delta.content, end="")
        print("\n\n")
else:
    while True:
        # Get the input
        query = input("User: ")

        # Generate the text
        print("\n\n")
        print(f"LLM_IC: {llm.generate_response(query)}", end="\n\n")
        # print(llm.generate_rag_text(query))
