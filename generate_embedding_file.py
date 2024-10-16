# importing required modules
from pypdf import PdfReader
import re
import pandas as pd
from langchain.text_splitter import NLTKTextSplitter
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

tqdm.pandas()

# Set the device to 'cuda' if you have a GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


CHUNK_SIZE = 1000

# creating a pdf reader object
reader = PdfReader(
    "2-Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.pdf"
)


BOOK_NAME = "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"

# Get the chapter titles


summary_df = pd.DataFrame(columns=["Chapter", "Title", "Page"])

for i in range(4, 13):
    page_text = reader.pages[i].extract_text()
    page_text = page_text.split("\n")
    # Get the ones that start with a number than a dot
    for line in page_text:
        if re.match(r"^\d+\.", line):
            chapter = re.findall(r"^\d+\.", line)[0]
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

            summary_df = summary_df._append(
                {"Chapter": chapter, "Title": title, "Page": page}, ignore_index=True
            )

# Get the text from all the chapters and save it to a df


text_splitter = NLTKTextSplitter(chunk_size=CHUNK_SIZE, separator="\n")
full_df = pd.DataFrame(columns=["Chapter", "Title", "Text"])

page_correction = 25

for i in tqdm(range(0, len(summary_df)), desc="Processing chapters", unit="chapter"):
    initial_page = summary_df.iloc[i]["Page"] + page_correction
    if i == len(summary_df) - 1:
        final_page = len(reader.pages)
    else:
        final_page = summary_df.iloc[i + 1]["Page"] + page_correction

    chapter_text = ""

    # print(f"Title {i+1}: {summary_df.iloc[i]['Title']}")

    for j in range(initial_page, final_page):
        page_text = reader.pages[j].extract_text()

        page_text = page_text.split("\n")

        # Get the ones with "26 | Chapter 1:"
        for line in page_text:
            if (
                re.match(r"^\d+ \| Chapter \d+:", line)
                or re.match(r"CHAPTER \d+", line)
                or re.match(r"With Early Release ebooks", line)
                or re.match(r"the author’s raw", line)
                or re.match(
                    r"can take advantage of these technologies long before the official",
                    line,
                )
                or re.match(r"release of these titles.", line)
                or re.match(r"release of the book.", line)
            ):
                pass
            else:
                # Encode line in UTF-8 and decode back to string
                chapter_text += line.encode("utf-8", errors="ignore").decode("utf-8")

    # Divide the text into chunks
    chapter_chunks = text_splitter.split_text(chapter_text)

    # print(f"Number of chunks: {len(chapter_chunks)}")

    for chunk in chapter_chunks:
        full_df = full_df._append(
            {
                "Chapter": summary_df.iloc[i]["Chapter"],
                "Title": summary_df.iloc[i]["Title"],
                "Text": chunk,
                "Book": BOOK_NAME,
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
full_df.to_json("full_df_embeddings.json", orient="records")
