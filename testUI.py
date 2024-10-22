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
root.geometry("800x600")  # Set the window size to 800x600

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
labels = []

# Populate the frame with checkboxes for each chapter and topic
for item in toc:
    chapter_var = tk.IntVar()
    topic_var = tk.IntVar()
    checkbox_vars[item[1]] = (chapter_var, topic_var)

    # Create only one entry for each item but with two checkboxes
    entry_frame = tk.Frame(checkbox_frame)
    entry_frame.pack(anchor="w", fill="x")

    # Create a label for the chapter/topic
    label = tk.Label(entry_frame, text=item[1])
    label.pack(side="left", padx=5)
    labels.append(label)

    # Create the checkboxes for chapter and topic
    chapter_checkbox = tk.Checkbutton(entry_frame, text="Chapter", variable=chapter_var)
    chapter_checkbox.pack(side="right", padx=5)

    topic_checkbox = tk.Checkbutton(entry_frame, text="Topic", variable=topic_var)
    topic_checkbox.pack(side="right", padx=5)


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


# Enable scrolling with the mouse wheel
def on_mouse_wheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


canvas.bind_all("<MouseWheel>", on_mouse_wheel)

# Navigation with arrow keys
current_label_index = 0


def highlight_label(index):
    for i, label in enumerate(labels):
        if i == index:
            label.config(bg="yellow")
        else:
            label.config(bg="white")


highlight_label(current_label_index)


def on_key_press(event):
    global current_label_index

    # Ensure the selected label is visible
    label = labels[current_label_index]
    canvas.update_idletasks()  # Update the canvas to get the correct height
    canvas.yview_moveto(max(0, min(label.winfo_y() / checkbox_frame.winfo_height(), 1)))

    if event.keysym == "Up":
        if current_label_index > 0:
            current_label_index -= 1
    elif event.keysym == "Down":
        if current_label_index < len(labels) - 1:
            current_label_index += 1
    elif event.keysym == "c":
        chapter_var, topic_var = checkbox_vars[labels[current_label_index].cget("text")]
        chapter_var.set(1 - chapter_var.get())
    elif event.keysym == "t":
        chapter_var, topic_var = checkbox_vars[labels[current_label_index].cget("text")]
        topic_var.set(1 - topic_var.get())
    elif event.keysym == "BackSpace":
        chapter_var, topic_var = checkbox_vars[labels[current_label_index].cget("text")]
        chapter_var.set(0)
        topic_var.set(0)

    highlight_label(current_label_index)


root.bind("<KeyPress>", on_key_press)

# Start the Tkinter event loop
root.mainloop()

# Print the selected chapters and topics
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
