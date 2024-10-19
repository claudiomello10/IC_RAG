# importing required modules
from pypdf import PdfReader
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from langchain.text_splitter import NLTKTextSplitter
from tqdm import tqdm

tqdm.pandas()

# Set the device to 'cuda' if you have a GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


# creating a pdf reader object
reader = PdfReader(
    "2-Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.pdf"
)


CHUNK_SIZE = 3000

# Get the chapter and section titles
summary_dict = {}

for i in range(4, 13):
    page_text = reader.pages[i].extract_text()
    page_text = page_text.split("\n")
    # Get the ones that start with a number than a dot
    current_chapter = ""
    for line in page_text:
        if re.match(r"^\d+\.", line):
            chapter = re.findall(r"^\d+\.", line)[0]
            chapter = int(chapter[:-1])
            title = re.sub(r"^\d+\.", "", line)
            # Get the page at the end of the chapter name
            page = re.findall(r"\d+$", title)[0]
            page = int(page)

            # Remove the dots from the title
            title = re.sub(r"\.", "", title)

            # Remove the page number from the title
            title = re.sub(r"\d+$", "", title)

            # Remove the spaces at the end of the title
            title = title.strip()

            summary_dict[chapter] = {"Title": title, "Page": page, "topics": []}

        else:
            # Match the ones that end with a number
            result = re.match(r".*\d+$", line)
            if result:
                topic_text = result.group(0)
                topic_text = topic_text.split("  ")
                topic = topic_text[0]
                page = topic_text[-1]
                page = int(page)
                summary_dict[chapter]["topics"].append({"Topic": topic, "Page": page})


# Get the text from all the chapters and save it to a df


text_splitter = NLTKTextSplitter(chunk_size=CHUNK_SIZE, separator="\n")
full_df = pd.DataFrame(columns=["Chapter", "Title", "Text", "Topic"])

page_correction = 25

for chapter in tqdm(summary_dict):
    title = summary_dict[chapter]["Title"]
    chapter_page = summary_dict[chapter]["Page"] + page_correction
    topics = summary_dict[chapter]["topics"]

    # Get the text from the chapter withouth a topic (Before the first topic)
    pre_topic_text = ""
    for i in range(chapter_page, topics[0]["Page"] + page_correction):
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
                "Chapter": chapter,
                "Title": title,
                "Text": text,
                "Topic": "Chapter Introduction",
            },
            ignore_index=True,
        )

    for topic in topics:
        topic_title = topic["Topic"]
        topic_page = topic["Page"] + page_correction
        # Get the page of the next topic, in case it is the last topic in the chapter we will use the next chapter page, in case it is the last chapter we will use the last page
        next_topic_title = ""
        if topic == topics[-1]:
            if chapter == 14:
                next_topic_page = len(reader.pages)
            else:
                next_topic_title = summary_dict[chapter + 1]["topics"][0]["Topic"]
                next_topic_page = summary_dict[chapter + 1]["Page"] + page_correction
        else:
            next_topic_page = topics[topics.index(topic) + 1]["Page"] + page_correction
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
                    "Chapter": chapter,
                    "Title": title,
                    "Text": text,
                    "Topic": topic_title,
                },
                ignore_index=True,
            )


# Create an embedding space for the chunks using
embedding_model = SentenceTransformer("BAAI/bge-large-en", device=device)


# Create the embeddings for the chunks
print("Creating embeddings for the chunks")
full_df["Embedding"] = full_df["Text"].progress_apply(
    lambda x: embedding_model.encode(x)
)

print("Saving the embeddings to a file")
full_df.to_json("full_df_embeddings_sections.json", orient="records")
