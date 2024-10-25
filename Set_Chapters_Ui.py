import tkinter as tk
import os
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
from pypdf import PdfReader
from langchain.text_splitter import NLTKTextSplitter
from tkinter import scrolledtext

# Set the tqdm to pandas
tqdm.pandas()


def get_summary_dict(path: str):

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    canceled = False

    # Step 3: Open the PDF file
    doc = fitz.open(path)

    # Step 4: Extract the bookmarks (table of contents)
    toc = doc.get_toc()

    # Create the main window
    root = tk.Tk()
    root.title("Select Chapter")
    root.geometry("800x600")  # Set the window size to 800x600

    # Create a header label
    header_label = tk.Label(
        root,
        text="Press space to select the chapters, backspace to erase and arrows to select",
        font=("Helvetica", 12, "bold"),
    )
    header_label.pack(pady=10)

    # Create a ScrolledText widget
    scrolled_list = scrolledtext.ScrolledText(root, wrap=tk.WORD)
    scrolled_list.pack(fill="both", expand=True)

    # Create a dictionary to hold the IntVar for each checkbox
    checkbox_vars = {}
    labels = []

    # Populate the ScrolledText with checkboxes for each chapter and topic
    for item in toc:
        chapter_var = tk.IntVar()
        checkbox_vars[item[1]] = chapter_var

        # Create only one entry for each item but with two checkboxes
        entry_frame = tk.Frame(scrolled_list)
        scrolled_list.window_create("end", window=entry_frame)
        scrolled_list.insert("end", "\n")

        # Create the checkboxes for chapter and topic
        chapter_checkbox = tk.Checkbutton(entry_frame, variable=chapter_var)
        chapter_checkbox.pack(side="left", padx=5)

        # Create a label for the chapter/topic
        label = tk.Label(entry_frame, text=item[1])
        label.pack(side="left", padx=5)
        labels.append(label)

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
                # Scroll the list to the current label
                scrolled_list.yview_moveto((current_label_index - 10) / len(labels))
        elif event.keysym == "Down":
            if current_label_index < len(labels) - 1:
                current_label_index += 1
                # Scroll the list to the current label
                scrolled_list.yview_moveto((current_label_index - 10) / len(labels))

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
            # close the window and return the selected chapters
            root.quit()
            root.destroy()

        highlight_label(current_label_index)

    root.bind("<KeyPress>", on_key_press)

    # If the user closes the window, set the canceled flag to True
    def on_closing():
        nonlocal canceled
        canceled = True
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the Tkinter event loop
    root.mainloop()

    # Print the selected chapters and topics
    selected_chapters = [
        chapter
        for chapter, chapter_var in checkbox_vars.items()
        if chapter_var.get() == 1
    ]

    # Based on the selected chapters and topics, create a summary dictionary
    if canceled:
        # Close the PDF document
        doc.close()

        return None
    else:
        summary_list = []
        training_list = []
        current_chapter = None
        for item in toc:
            if item[1] in selected_chapters:
                # Create a new entry in the summary list
                summary_list.append({"Page": item[2], "Title": item[1], "topics": []})
                current_chapter = item[1]
                # Add the chapter to the training list
                training_list.append(
                    {
                        "Name": item[1],
                        "Importance_index": item[0],
                        "Page": item[2],
                        "IsChapter": True,
                    }
                )

            else:
                if current_chapter is not None:
                    # Add the topic to the current chapter
                    summary_list[-1]["topics"].append(
                        {"Page": item[2], "Topic": item[1]}
                    )
                # Add the topic to the training list
                training_list.append(
                    {
                        "Name": item[1],
                        "Importance_index": item[0],
                        "Page": item[2],
                        "IsChapter": False,
                    }
                )

        # Close the PDF document
        doc.close()

        return summary_list, training_list


def get_chunks(path: str, chunk_size: int = 3000):

    # Get book name
    book_name = os.path.basename(path)

    # Get the summary dictionary
    summary_list, training_list = get_summary_dict(path)

    # creating a pdf reader object
    reader = PdfReader(path)

    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, separator="\n")

    full_df = pd.DataFrame(columns=["Chapter", "Text", "Topic"])

    for index, chapter in tqdm(enumerate(summary_list)):
        title = chapter["Title"]
        chapter_page = chapter["Page"] - 1
        topics = chapter["topics"]

        # Get the text from the chapter withouth a topic (Before the first topic)
        pre_topic_text = ""
        for i in range(chapter_page, topics[0]["Page"]):
            page_text = reader.pages[i].extract_text()
            # Check if the topic title is in the page
            title_test_text = topics[0]["Topic"] + "\n"
            if title_test_text in page_text:
                # Get the text after the title
                page_text = page_text.split(title_test_text)[0]
            pre_topic_text += page_text
        pre_topic_text = text_splitter.split_text(pre_topic_text)
        for text in pre_topic_text:
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
            full_df = full_df._append(
                {
                    "Book": book_name,
                    "Chapter": title,
                    "Text": text,
                    "Topic": "Chapter Introduction",
                },
                ignore_index=True,
            )

        for topic in topics:
            topic_title = topic["Topic"]
            topic_page = topic["Page"] - 1
            # Get the page of the next topic, in case it is the last topic in the chapter we will use the next chapter page, in case it is the last chapter we will use the last page
            next_topic_title = ""
            if topic == topics[-1]:
                if chapter == summary_list[-1]:
                    next_topic_page = len(reader.pages)
                else:
                    next_topic_title = summary_list[index + 1]["topics"][0]["Topic"]
                    next_topic_page = summary_list[index + 1]["Page"]
            else:
                next_topic_page = topics[topics.index(topic) + 1]["Page"]
                next_topic_title = topics[topics.index(topic) + 1]["Topic"]

            topic_text = ""
            # Get the text until the next topic text or the next chapter text (if it is the last topic) appears
            for i in range(topic_page, next_topic_page):

                page_text = reader.pages[i].extract_text()
                # find the current topic title
                title_test_text = topic_title + "\n"
                if title_test_text in page_text:
                    # Get the text after the title
                    page_text = page_text.split(title_test_text)[1]

                # Check if the topic title is in the page
                title_test_text = next_topic_title + "\n"
                if title_test_text in page_text:
                    # Get the text before the next topic title
                    page_text = page_text.split(title_test_text)[0]
                topic_text += page_text
            topic_text = text_splitter.split_text(topic_text)
            for text in topic_text:
                text = text.encode("utf-8", errors="ignore").decode("utf-8")
                full_df = full_df._append(
                    {
                        "Book": book_name,
                        "Chapter": title,
                        "Text": text,
                        "Topic": topic_title,
                    },
                    ignore_index=True,
                )
    return full_df


def get_chunks_with_train_data(path: str, chunk_size: int = 3000):

    # Get book name
    book_name = os.path.basename(path)

    # Get the summary dictionary
    summary_list, training_list = get_summary_dict(path)

    training_df = pd.DataFrame(training_list)

    # creating a pdf reader object
    reader = PdfReader(path)

    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, separator="\n")

    full_df = pd.DataFrame(columns=["Chapter", "Text", "Topic"])

    for index, chapter in tqdm(enumerate(summary_list)):
        title = chapter["Title"]
        chapter_page = chapter["Page"] - 1
        topics = chapter["topics"]

        # Get the text from the chapter withouth a topic (Before the first topic)
        pre_topic_text = ""
        for i in range(chapter_page, topics[0]["Page"]):
            page_text = reader.pages[i].extract_text()
            # Check if the topic title is in the page
            title_test_text = topics[0]["Topic"] + "\n"
            if title_test_text in page_text:
                # Get the text after the title
                page_text = page_text.split(title_test_text)[0]
            pre_topic_text += page_text
        pre_topic_text = text_splitter.split_text(pre_topic_text)
        for text in pre_topic_text:
            text = text.encode("utf-8", errors="ignore").decode("utf-8")
            full_df = full_df._append(
                {
                    "Book": book_name,
                    "Chapter": title,
                    "Text": text,
                    "Topic": "Chapter Introduction",
                },
                ignore_index=True,
            )

        for topic in topics:
            topic_title = topic["Topic"]
            topic_page = topic["Page"] - 1
            # Get the page of the next topic, in case it is the last topic in the chapter we will use the next chapter page, in case it is the last chapter we will use the last page
            next_topic_title = ""
            if topic == topics[-1]:
                if chapter == summary_list[-1]:
                    next_topic_page = len(reader.pages)
                else:
                    next_topic_title = summary_list[index + 1]["topics"][0]["Topic"]
                    next_topic_page = summary_list[index + 1]["Page"]
            else:
                next_topic_page = topics[topics.index(topic) + 1]["Page"]
                next_topic_title = topics[topics.index(topic) + 1]["Topic"]

            topic_text = ""
            # Get the text until the next topic text or the next chapter text (if it is the last topic) appears
            for i in range(topic_page, next_topic_page):

                page_text = reader.pages[i].extract_text()
                # find the current topic title
                title_test_text = topic_title + "\n"
                if title_test_text in page_text:
                    # Get the text after the title
                    page_text = page_text.split(title_test_text)[1]

                # Check if the topic title is in the page
                title_test_text = next_topic_title + "\n"
                if title_test_text in page_text:
                    # Get the text before the next topic title
                    page_text = page_text.split(title_test_text)[0]
                topic_text += page_text
            topic_text = text_splitter.split_text(topic_text)
            for text in topic_text:
                text = text.encode("utf-8", errors="ignore").decode("utf-8")
                full_df = full_df._append(
                    {
                        "Book": book_name,
                        "Chapter": title,
                        "Text": text,
                        "Topic": topic_title,
                    },
                    ignore_index=True,
                )
    # Close the PDF document
    reader.close()

    return full_df, training_df
