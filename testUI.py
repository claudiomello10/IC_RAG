import tkinter as tk
from tkinter import ttk

import fitz  # PyMuPDF

# Step 2: Define the path to the PDF file
PATH = "Livros/Futuros/MachineLearning/2024_Python_Deep_Learning_w_pacc21.pdf"

# Step 3: Open the PDF file
doc = fitz.open(PATH)

# Step 4: Extract the bookmarks (table of contents)
toc = doc.get_toc()

# Step 5: Print the bookmarks
for item in toc:
    print(item)

# Create the main window
root = tk.Tk()
root.title("Select Chapter")

# Create a frame to hold the listbox and checkboxes
frame = tk.Frame(root)
frame.pack(side="left", fill="both", expand=True)

# Create a canvas to allow scrolling
canvas = tk.Canvas(frame)
canvas.pack(side="left", fill="both", expand=True)

# Add a scrollbar to the canvas
scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
scrollbar.pack(side="right", fill="y")

# Configure the canvas to work with the scrollbar
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create another frame inside the canvas to hold the checkboxes
checkbox_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")

# Create a dictionary to hold the IntVar for each checkbox
checkbox_vars = {}

# Populate the frame with checkboxes for each chapter and topic
for item in toc:
    chapter_var = tk.IntVar()
    topic_var = tk.IntVar()
    checkbox_vars[item[1]] = (chapter_var, topic_var)

    # Create only one entry for each item but with two checkboxes
    entry_frame = tk.Frame(checkbox_frame)
    entry_frame.pack(anchor="w", fill="x")

    chapter_checkbox = tk.Checkbutton(
        entry_frame, text=f"Chapter: {item[1]}", variable=chapter_var
    )
    chapter_checkbox.pack(side="left")

    topic_checkbox = tk.Checkbutton(
        entry_frame, text=f"Topic: {item[1]}", variable=topic_var
    )
    topic_checkbox.pack(side="left")


# Function to get selected chapters and topics
def on_select():
    selected_chapters = [
        chapter
        for chapter, (chapter_var, topic_var) in checkbox_vars.items()
        if chapter_var.get() == 1
    ]
    selected_topics = [
        topic
        for topic, (chapter_var, topic_var) in checkbox_vars.items()
        if topic_var.get() == 1
    ]
    print(f"Selected chapters: {selected_chapters}")
    print(f"Selected topics: {selected_topics}")


# Add a button to print the selected chapters and topics
select_button = tk.Button(root, text="Select", command=on_select)
select_button.pack(side="bottom")

# Start the Tkinter event loop
root.mainloop()
