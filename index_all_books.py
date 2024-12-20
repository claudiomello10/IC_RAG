from Set_Chapters_Ui import get_chunks_with_train_data
import os
import pandas as pd
from tqdm import tqdm

books_dir = "Livros/All"

# Get all the books in the directory
books = os.listdir(books_dir)


all_chunk_df = (
    pd.read_csv("all_chunks.csv")
    if os.path.exists("all_chunks.csv")
    else pd.DataFrame()
)
all_train_df = (
    pd.read_csv("all_training.csv")
    if os.path.exists("all_training.csv")
    else pd.DataFrame()
)

pb = tqdm(books, total=len(books), desc="Books", unit="Book")

for book in pb:
    try:
        pb.set_description(f"Processing {book}")
        # Get the path to the book
        book_path = os.path.join(books_dir, book)

        # Get the chunk list and training list
        chunk_df, training_df = get_chunks_with_train_data(book_path)

        # Concatenate the dataframes
        all_chunk_df = pd.concat([all_chunk_df, chunk_df], ignore_index=True)
        all_train_df = pd.concat([all_train_df, training_df], ignore_index=True)

        # Save the dataframes
        all_chunk_df.to_csv("all_chunks.csv", index=False)
        all_train_df.to_csv("all_training.csv", index=False)

        # Ensure the Completed directory exists
        completed_dir = "Livros/Completed"
        if not os.path.exists(completed_dir):
            os.makedirs(completed_dir)

        # Change the location of the file to a Completed folder
        os.rename(book_path, os.path.join(completed_dir, book))
    except Exception as e:
        print(f"Error processing book {book}: {e}")
        continue


print("All chunks and training data saved")
