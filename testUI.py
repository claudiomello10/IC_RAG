import tkinter as tk
import os
import fitz  # PyMuPDF

# Step 2: Define the path to the PDF file
PATH = "Livros/Futuros/MachineLearning/2019_Book_EmbeddedDeepLearning.pdf"


def get_summary_dict(path: str):

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    # Step 3: Open the PDF file
    doc = fitz.open(path)

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
    canvas.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # Create another frame inside the canvas to hold the checkboxes
    checkbox_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=checkbox_frame, anchor="nw")

    # Create a header label
    header_label = tk.Label(
        checkbox_frame,
        text="Press space to select topic, c for chapter, backspace to erase and arrows to select",
        font=("Helvetica", 12, "bold"),
    )
    header_label.pack(pady=10)

    # Create a dictionary to hold the IntVar for each checkbox
    checkbox_vars = {}
    labels = []

    # Populate the frame with checkboxes for each chapter and topic
    for item in toc:
        chapter_var = tk.IntVar()
        checkbox_vars[item[1]] = chapter_var

        # Create only one entry for each item but with two checkboxes
        entry_frame = tk.Frame(checkbox_frame)
        entry_frame.pack(anchor="w", fill="x")

        # Create a label for the chapter/topic
        label = tk.Label(entry_frame, text=item[1])
        label.pack(side="left", padx=5)
        labels.append(label)

        # Create the checkboxes for chapter and topic
        chapter_checkbox = tk.Checkbutton(
            entry_frame, text="Chapter", variable=chapter_var
        )
        chapter_checkbox.pack(side="right", padx=5)

    # Function to get selected chapters and topics
    def on_select():
        selected_chapters = [
            chapter
            for chapter, chapter_var in checkbox_vars.items()
            if chapter_var.get() == 1
        ]
        print(f"Selected chapters: {selected_chapters}")

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
        nonlocal current_label_index

        if event.keysym == "Up":
            if current_label_index > 0:
                current_label_index -= 1
        elif event.keysym == "Down":
            if current_label_index < len(labels) - 1:
                current_label_index += 1
        elif event.keysym == "space":
            chapter_var = checkbox_vars[labels[current_label_index].cget("text")]
            chapter_var.set(1 - chapter_var.get())
        elif event.keysym == "BackSpace":
            chapter_var = checkbox_vars[labels[current_label_index].cget("text")]
            chapter_var.set(0)
        elif event.keysym == "Delete":
            for chapter_var in checkbox_vars.values():
                chapter_var.set(0)
        elif event.keysym == "Return":
            # close the window
            root.quit()

        highlight_label(current_label_index)

    root.bind("<KeyPress>", on_key_press)

    # Start the Tkinter event loop
    root.mainloop()

    # Print the selected chapters and topics
    selected_chapters = [
        chapter
        for chapter, chapter_var in checkbox_vars.items()
        if chapter_var.get() == 1
    ]

    # Based on the selected chapters and topics, create a summary dictionary
    summary_dict = {}
    current_chapter = None
    for item in toc:
        if item[1] in selected_chapters:
            summary_dict[item[1]] = {"Page": item[0], "topics": []}
            current_chapter = item[1]
        else:
            if current_chapter is not None:
                summary_dict[current_chapter]["topics"].append(
                    {"Page": item[0], "Topic": item[1]}
                )

    return summary_dict


summary_dict = get_summary_dict(PATH)

print(summary_dict)
