import os
import warnings

# Remove the warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Import the LLM_IC class
from LLM_IC import LLM_IC
import tkinter as tk
from tkinter import scrolledtext


# Create the LLM_IC object
llm = LLM_IC(
    embeddings_paths=[
        "full_df_embeddings_sections1.json",
        "full_df_embeddings_sections2.json",
    ]
)


def send_query():
    query = user_input.get()
    user_input.set("")  # Clear the input field
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, f"User: {query}\n\n", "user_query")  # Added line break
    chat_window.config(state=tk.DISABLED)

    response = llm.generate_response_stream(query)
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "LLM_IC: ")
    chat_window.config(state=tk.DISABLED)

    def stream_response():
        for r in response:
            if r.choices[0].delta.content is not None:
                content = r.choices[0].delta.content
                if "**" in content:
                    parts = content.split("**")
                    for i, part in enumerate(parts):
                        chat_window.config(state=tk.NORMAL)
                        if i % 2 == 1:
                            chat_window.insert(tk.END, part, "bold")
                        else:
                            chat_window.insert(tk.END, part)
                        chat_window.config(state=tk.DISABLED)
                else:
                    chat_window.config(state=tk.NORMAL)
                    chat_window.insert(tk.END, content)
                    chat_window.config(state=tk.DISABLED)
                chat_window.yview(tk.END)
            root.update_idletasks()
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "\n\n")  # Added line break after LLM response
        chat_window.config(state=tk.DISABLED)

    root.after(0, stream_response)


# Create the main window
root = tk.Tk()
root.title("Chatbot")

# Create a frame for the chat window
chat_frame = tk.Frame(root)
chat_frame.pack(padx=10, pady=10)

# Create a scrolled text widget for the chat window
chat_window = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, state=tk.DISABLED)
chat_window.pack(padx=10, pady=10)

# Add tags for bold text and user query
chat_window.tag_configure("bold", font=("TkDefaultFont", 10, "bold"))
chat_window.tag_configure("user_query", foreground="blue")

# Create a frame for the user input
input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=10)

# Create a StringVar to hold the user input
user_input = tk.StringVar()

# Create an entry widget for the user input
input_entry = tk.Entry(input_frame, textvariable=user_input, width=50)
input_entry.pack(side=tk.LEFT, padx=10, pady=10)

# Create a button to send the query
send_button = tk.Button(input_frame, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=10, pady=10)

# Start the main event loop
root.mainloop()
