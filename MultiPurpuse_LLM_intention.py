import pandas as pd
import faiss
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import ujson as json
from openai import OpenAI
import os
import fitz
from pypdf import PdfReader
from langchain.text_splitter import NLTKTextSplitter
from tqdm import tqdm
from transformers import pipeline


class StudentHelper:
    """
    The StudentHelper class provides a set of methods to help students study and learn from books.
    The class uses a combination of embeddings and language models to provide answers to student questions.
    The class can be used to generate embeddings from text data, create a FAISS index, search for similar embeddings,
    generate retrieval-augmented generation (RAG) text, and generate responses to student questions.

    Attributes:
    device (str): The device to run the model on (CPU or GPU).
    embeddings_model (str): The SentenceTransformer model to use for generating embeddings.
    model (str): The language model to use for generating responses.
    messages (list): The list of messages in the conversation.
    embeddings_path (str): The path to the embeddings file.
    embedding_df (pd.DataFrame): The DataFrame containing the text data with embeddings.
    faiss_index (faiss.IndexFlatIP): The FAISS index for the embeddings.

    Methods:
    load_embeddings(embeddings_path: str): Load the embeddings DataFrame from a file.
    create_faiss_index(): Create a FAISS index from the embeddings DataFrame.
    save_embeddings(embeddings_path: str): Save the embeddings DataFrame to a file.
    create_faiss_index_from_embeddings_file(embeddings_path: str): Create a FAISS index from the embeddings file.
    update_faiss_index(): Update the FAISS index with the given embeddings.
    search(query: str, top_k: int = 5): Search for the top_k most similar embeddings to the query and return the corresponding DataFrame rows.
    generate_embeddings(query: str): Generate embeddings for the given query using the embeddings model.
    decode(embedding: np.array): Decode the given embedding back to text using the embeddings model.
    generate_rag_text(query: str, top_k: int = 5): Generate a retrieval-augmented
    generation (RAG) text based on the query and top_k search results.
    generate_response(query: str, model: str | None = None): Generate a response to the query using the language model and RAG context.
    generate_response_conversation(query: str, model: str | None = None): Generate a response to the query using the language model, RAG context, and conversation history.
    generate_response_stream(query: str, model: str | None = None): Generate a streaming response to the query using the language model and RAG context.
    get_table_of_contents_from_PDF(path: str): Get the table of contents (TOC) from a PDF file.
    get_processed_name(name: str): Returns the processed name by making it lowercase, removing leading and trailing spaces, removing sequences of spaces and newlines.
    generate_toc_text_for_prompting(toc: list, book_name: str): Generate a prompt text from the table of contents (TOC).
    get_model_answer_of_chapters(toc_prompt: str, model: str): Get the model answer of chapters from the table of contents.
    get_summary_list_from_PDF(path: str, book_name=None): Get the summary list from a PDF file.
    get_chunks_df_from_PDF(path: str, book_name=None, chunk_size=3000): Get the chunks DataFrame from a PDF file.
    get_embeddings_df(df: pd.DataFrame): Get the embeddings DataFrame from the given DataFrame.
    get_embeddings_df_from_PDF(path: str, book_name=None, chunk_size=3000): Get the embeddings DataFrame from the given PDF file.
    get_chunks_for_multiple_PDFs(paths: list, book_names: list = None, chunk_size=3000): Get the text chunks for multiple PDF files.
    get_embeddings_df_from_multiple_PDFs(paths: list, book_names: list = None, chunk_size=3000): Get the embeddings DataFrame for multiple PDF files.
    set_embeddings_df(df: pd.DataFrame): Set the embeddings DataFrame.
    set_embeddings_df_from_PDF(path: str, book_name=None, chunk_size=3000, update_index=True): Set the embeddings DataFrame from the given PDF file.
    set_embeddings_df_from_multiple_PDFs(paths: list, book_names: list = None, chunk_size=3000, update_index=True): Set the embeddings DataFrame for multiple PDF files.
    add_PDF_to_embeddings_df(path: str, book_name=None, chunk_size=3000, update_index=True): Add the text data from the given PDF file to the embeddings DataFrame.
    add_multiple_PDFs_to_embeddings_df(paths: list, book_names: list = None, chunk_size=3000, update_index=True): Add the text data from multiple PDF files to the embeddings DataFrame.
    """

    def __init__(
        self,
        device: str = None,
        embeddings_model: str = "jinaai/jina-embeddings-v3",
        model: str = "gpt-4o-mini",
    ):
        """
        Initializes the LLM_IC class with the given parameters and loads the embeddings and models.
        Args:
            device (str): The device to run the model on (CPU or GPU).
            embeddings_model (str): The SentenceTransformer model to use for generating embeddings.
            model (str): The language model to use for generating responses.
        Returns:
            StudentHelper object.
        """

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model
        self.load_prompt_classifier()
        self.messages = []
        # Define the classes
        self.classes = [
            "search query",
            "pergunta sobre um assunto",
            "pedido de definição",
            "pedido de indicação",
        ]

        # Create a map from the classes to numbers, 1 for searching and 2 for asking
        self.class_map = {
            "search query": 1,
            "pergunta sobre um assunto": 2,
            "pedido de definição": 3,
            "pedido de indicação": 1,
        }

        # Create a map from the numbers to objectives
        self.objective_map = {1: "search", 2: "ask", 3: "definition"}

        # Load the embeddings model
        try:
            print("Loading the embeddings model")
            self.embeddings_model = SentenceTransformer(
                embeddings_model, device=device, trust_remote_code=True
            )
            print("Embeddings model loaded")
        except Exception as e:
            print(f"Error loading the embeddings model: {e}")

        self.client = OpenAI()

    def load_prompt_classifier(self, prompt_classifier="facebook/bart-large-mnli"):
        """
        Load the prompt classifier model.
        Args:
            prompt_classifier (str): The prompt classifier model to use.
        """
        print("Loading the prompt classifier model")
        self.prompt_classifier = pipeline(
            "zero-shot-classification", model=prompt_classifier, device=self.device
        )
        print("Prompt classifier model loaded")

    def get_intent(self, text: str):
        """
          There are 3 possible intents: search, ask, and definition.
        Args:
            text (str): The text to classify.
        Returns:
            str: The intent of the query. Possible values are 'search', 'ask', and 'definition'.
        """

        result = self.prompt_classifier(text, self.classes, multi_class=True)
        # Get the highest score
        max_score = max(result["scores"])

        # Get the class with the highest score
        max_class = result["labels"][result["scores"].index(max_score)]

        # Get the objective
        objective = self.objective_map[self.class_map[max_class]]

        return objective

    def set_subject(self, subject: str):
        """
        Set the subject for the student helper.
        Args:
            subject (str): The subject to set.
        """
        self.subject = subject

    def load_embeddings(self, embeddings_path: str):

        try:
            if embeddings_path:
                # Raise error if the embeddings path does not exist
                embedding_list = []

                if not os.path.exists(embeddings_path):
                    raise FileNotFoundError(
                        f"Embeddings file not found at {embeddings_path}"
                    )
                with open(embeddings_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Create a list of dictionaries

                for item in data:
                    embedding_list.append(
                        {
                            "Chapter": item["Chapter"],
                            "Text": item["Text"],
                            "Embedding": np.array(item["Embedding"]),
                            "Topic": item["Topic"],
                            "Book": item["Book"],
                        }
                    )

            self.embeddings_path = embeddings_path
            # Create a dataframe
            self.embedding_df = pd.DataFrame(embedding_list)
            print("Embeddings loaded successfully")
        except Exception as e:
            raise e

    def create_faiss_index(self):
        if self.embedding_df is not None:
            # Create a faiss index
            try:
                # Get the embeddings dimension
                embeddings_dimension = np.array(
                    self.embedding_df["Embedding"][0]
                ).shape[0]

                # Create a faiss index
                self.faiss_index = faiss.IndexFlatIP(embeddings_dimension)

                # Add the embeddings to the index
                embeddings = np.array(self.embedding_df["Embedding"].to_list())

                # Check if the embeddings are 1D
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                # Add the embeddings to the faiss index
                self.faiss_index.add(embeddings)

            except Exception as e:
                print(f"Error creating the faiss index: {e}")
        else:
            raise Exception("Embeddings DataFrame is not loaded")

    def save_embeddings(self, embeddings_path: str):
        """
        Save the embeddings DataFrame to a file.
        Args:
            embeddings_path (str): The path to save the embeddings to.
        """
        self.embedding_df.to_json(embeddings_path, orient="records")

    def create_faiss_index_from_embeddings_file(self, embeddings_path: str):
        """
        Create a FAISS index from the embeddings file.
        Args:
            embeddings_path (str): The path to the embeddings file.
        """
        self.load_embeddings(embeddings_path)
        self.create_faiss_index()

    def update_faiss_index(self):
        """
        Update the FAISS index with the given embeddings.
        Args:
            embeddings (np.array): The embeddings to update the index with.
        """
        if hasattr(self, "faiss_index"):
            embedding_list = self.embedding_df["Embedding"].to_list()
            embeddings = np.array(embedding_list)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            self.faiss_index.add()
            # Check if the embeddings are 1D

        else:
            raise Exception("FAISS index not created. Please create the FAISS index.")

    def search(self, query: str, top_k: int = 5):
        """
        Searches for the top_k most similar embeddings to the query and returns the corresponding DataFrame rows.
        Args:
            query (str): The query to search for.
            top_k (int): The number of top results to return.
        Returns:
            pd.DataFrame: The DataFrame containing the top_k search results.
        """
        if not hasattr(self, "faiss_index"):
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )

        query_embedding = self.generate_embeddings(query)
        query_embedding = query_embedding.reshape(1, -1)
        _, indices = self.faiss_index.search(query_embedding, top_k)
        return self.embedding_df.iloc[indices[0]]

    def generate_embeddings(self, query: str):
        """
        Generates embeddings for the given query using the embeddings model.
        Args:
            query (str): The query to generate embeddings for.
        Returns:
            np.array: The embeddings for the query.
        """

        embeddings = self.embeddings_model.encode(query)
        embeddings = np.array(embeddings)
        return embeddings

    def decode(self, embedding: np.array):
        """ "
        Decodes the given embedding back to text using the embeddings model.
        Args:
            embedding (np.array): The embedding to decode.
        Returns:
            str: The decoded text from the embedding.
        """
        return self.embeddings_model.decode(embedding)

    def generate_rag_text(self, query: str, intention: str, top_k: int = 5):
        """
        Generates a retrieval-augmented generation (RAG) text based on the query and top_k search results.
        Args:
            query (str): The query to generate RAG text for.
            top_k (int): The number of top results to use in the RAG text.
        Returns:
            str: The RAG text generated based on the query and search results.
        """
        if not hasattr(self, "faiss_index"):
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )

        # Verify if there is a subject
        if not hasattr(self, "subject"):
            self.subject = ""

        if intention == "ask":
            return self.generate_questions_rag_text(query, self.subject, top_k)
        elif intention == "search":
            return "SEARCH"
        elif intention == "definition":
            return "DEFINITION"

    def generate_questions_rag_text(self, query: str, subject, top_k: int = 5):
        """
        Generates a retrieval-augmented generation (RAG) text based on the query and top_k search results. This is used for question answering.
        Args:
            query (str): The query to generate RAG text for.
            top_k (int): The number of top results to use in the RAG text.
        Returns:
            str: The RAG text generated based on the query and search results.
        """
        rag_text = f"""You are an assistant helping a student to study {subject}.
        The student asks you a question and you provide an answer and an indication on which books and sections from the retrieval-augmented generation (RAG) context he can learn more about the topics, give the names chapters and sections of the books that should help him.
        When giving him the name of the book, you should provide the full name of the book.
        When giving him the name of the chapter, you should provide the full name of the chapter.
        When giving him the name of the section, you should provide the full name of the section.
        The citation should include the chapter and section of the book that was used to generate the answer.
        If the question is about a specific topic in the books, cite the chapter and section that defines the topic.
        If the student asks you a question that requires mathematical calculations do not provide the answer, provide only the method to solve the problem step by step, and instruct him where to find the solution in the book.
        Here is the context for the user query retrieved from the books:

        """
        search_results = self.search(query, top_k)
        retrieval_count = 1
        for index, row in search_results.iterrows():
            rag_text += f"Retriaval {retrieval_count}: From Book {row['Book']} - From Chapter {row['Chapter']} - Section: {row['Topic']}\n{row['Text']}\n\n"
            retrieval_count += 1
        return rag_text

    def generate_response(self, query: str, model: str | None = None):
        """
        Generates a response to the query using the language model and RAG context.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            str: The response generated by the language model.
        """
        if not hasattr(self, "faiss_index"):
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )

        if model is None:
            model = self.model

        intension = self.get_intent(query)

        rag_context = self.generate_rag_text(query, intension)
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        # Get OPENAI_API_KEY from the environment
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def generate_response_conversation(self, query: str, model: str | None = None):
        """
        Generates a response to the query using the language model, RAG context, and conversation history.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            str: The response generated by the language model.
        """
        if not hasattr(self, "faiss_index"):
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )

        if model is None:
            model = self.model
        rag_context = self.generate_rag_text(query)
        self.messages.append({"role": "system", "content": rag_context})
        self.messages.append({"role": "user", "content": query})
        messages = self.messages
        # Get OPENAI_API_KEY from the environment
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        self.messages.append(
            {"role": "assistant", "content": completion.choices[0].message.content}
        )
        return completion.choices[0].message.content

    def generate_response_stream(self, query: str, model: str | None = None):
        """
        Generates a streaming response to the query using the language model and RAG context.
        Args:
            query (str): The query to generate a response for.
            model (str): The language model to use for generating the response.
        Returns:
            response: The streaming response generated by the language model.
        """

        if not hasattr(self, "faiss_index"):
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )

        if model is None:
            model = self.model
        rag_context = self.generate_rag_text(query)
        messages = [
            {"role": "system", "content": rag_context},
            {"role": "user", "content": query},
        ]
        # Get OPENAI_API_KEY from the environment
        response = self.client.chat.completions.create(
            model=model, messages=messages, stream=True
        )
        return response

    def get_table_of_contents_from_PDF(self, path: str):
        """
        Get the table of contents (TOC) from a PDF file.
        Args:
            path (str): The path to the PDF file.
        Returns:
            toc (list): The table of contents extracted from the PDF file.
        """

        # Check if the file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file {path} does not exist.")

        # Step 3: Open the PDF file
        doc = fitz.open(path)

        # Step 4: Extract the bookmarks (table of contents)
        toc = doc.get_toc()

        doc.close()

        return toc

    def get_processed_name(self, name: str):
        """
        Returns the processed name by making it lowercase, removing leading and trailing spaces, removing sequences of spaces and newlines.
        Args:
            name (str): The name to process.
        Returns:
            str: The processed name.
        """
        # Make the string lowercase, remove any leading or trailing spaces, remove any sequence of spaces for one space and remove any sequence of newlines for one newline
        name = name.lower().strip().replace("\n", " ").replace("  ", " ")

        # Substitute all \xa0 with a space
        name = name.replace("\xa0", " ")

        return name

    def generate_toc_text_for_prompting(self, toc: list, book_name: str):
        """
        Generate a prompt text from the table of contents (TOC).
        Args:
            toc (list): The table of contents extracted from the PDF file.
        Returns:
            str: The prompt text generated from the table of contents.
        """

        # Initialize the prompt text
        text = "Book: " + book_name + "\n\nTable of Contents:\n\n"

        # Iterate over the table of contents
        for item in toc:
            # Get the title and page number
            title = item[1]
            page = item[2]
            importance_index = item[0]

            text += (
                "Importance Index: "
                + str(importance_index)
                + " -- "
                + "Topic name: "
                + str(title)
                + " -- "
                + "Topic page: "
                + str(page)
                + "\n"
            )

        return text

    def get_model_answer_of_chapters(self, toc_prompt: str, model: str):
        """
        Get the model answer of chapters from the table of contents.
        Args:
            toc_prompt (str): The prompt text generated from the table of contents.
            model (str): The language model to use for generating the response.
            book_name (str): The name of the book.
        Returns:
            str: The response generated by the language model.
        """

        prompt_engineering = """Given the table of contents of this book containing the importance index, topic name and topic page, provide me with the list of chapters and apendixes in the book.
        Do not inclue figures, tables, preface, index, bibliography,  or any other non-chapter or non-appendix sections.
        If an appendix is present with sub-sections, the sub-sections should not be included in the list, only the appendix name. For example, if the appendix is "Appendix A" and it has sub-sections "A.1", "A.2", "A.3", only "Appendix A" should be included in the list.
        If the book is separated into parts, the parts should not be included in the list.
        This list must be contain only the Name of the sections.
        The Names must be exactly as they appear in the table of contents.
        The response must have only the list in python format. For example, if the list is ['a', 'b', 'c'], the response must be ['a', 'b', 'c']. It cannot have any other text. If the list is empty, the response must be [].
        """

        full_prompt = toc_prompt + prompt_engineering

        messages = [{"role": "user", "content": full_prompt}]

        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        answer = completion.choices[0].message.content

        try:
            answer = eval(answer)
            return answer
        except:
            try:
                # Find the text between ```python and ``` and use that as the answer
                start = answer.find("```python")
                end = answer.find("```", start + 1)
                answer = answer[start + 9 : end].strip()
                answer = eval(answer)
                return answer
            except:
                raise Exception(
                    "Error in getting the chapters from the table of contents"
                )

    def get_summary_list_from_PDF(self, path: str, book_name=None):

        try:
            toc = self.get_table_of_contents_from_PDF(path)
        except Exception as e:
            raise e

        if book_name is None:
            # Get the book name from the path
            book_name = os.path.basename(path).split(".")[0]

        toc_prompt = self.generate_toc_text_for_prompting(toc, book_name)

        try:
            selected_chapters = self.get_model_answer_of_chapters(
                toc_prompt, self.model
            )
        except Exception as e:
            raise e

        # Process the selected chapters
        selected_chapters = [
            self.get_processed_name(chapter) for chapter in selected_chapters
        ]

        summary_list = []
        current_chapter = None
        for item in toc:
            processed_name = self.get_processed_name(item[1])

            if processed_name in selected_chapters:
                # Create a new entry in the summary list
                summary_list.append({"Page": item[2], "Title": item[1], "topics": []})
                current_chapter = item[1]

            else:
                if current_chapter is not None:
                    # Add the topic to the current chapter
                    summary_list[-1]["topics"].append(
                        {"Page": item[2], "Topic": item[1]}
                    )

        return summary_list

    def get_chunks_df_from_PDF(self, path: str, book_name=None, chunk_size=3000):

        if book_name is None:
            # Get the book name from the path
            book_name = os.path.basename(path).split(".")[0]

        try:
            summary_list = self.get_summary_list_from_PDF(path, book_name)
        except Exception as e:
            raise e

        # creating a pdf reader object
        reader = PdfReader(path)

        text_splitter = NLTKTextSplitter(chunk_size=chunk_size, separator="\n")

        full_df = pd.DataFrame(columns=["Chapter", "Text", "Topic"])

        for index, chapter in tqdm(
            enumerate(summary_list),
            desc=f"Processing chapters of book: {book_name}",
            total=len(summary_list),
            unit="chapter",
        ):
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

    def get_embeddings_df(self, df: pd.DataFrame):
        """
        Get the embeddings DataFrame from the given DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing the text data.
        Returns:
            pd.DataFrame: The DataFrame containing the text data with embeddings.
        """
        embeddings_list = []
        embedding_pb = tqdm(df.iterrows(), desc="Generating embeddings", total=len(df))

        for index, row in embedding_pb:
            text = row["Text"]
            embedding = self.generate_embeddings(text)
            embeddings_list.append(
                {
                    "Book": row["Book"],
                    "Chapter": row["Chapter"],
                    "Text": text,
                    "Embedding": embedding,
                    "Topic": row["Topic"],
                }
            )
        return pd.DataFrame(embeddings_list)

    def get_embeddings_df_from_PDF(self, path: str, book_name=None, chunk_size=3000):
        """
        Get the embeddings DataFrame from the given PDF file.
        Args:
            path (str): The path to the PDF file.
            book_name (str): The name of the book.
            chunk_size (int): The size of the text chunks to split the text into.
        Returns:
            pd.DataFrame: The DataFrame containing the text data with embeddings.
        """
        try:
            df = self.get_chunks_df_from_PDF(path, book_name, chunk_size)
        except Exception as e:
            raise e
        return self.get_embeddings_df(df)

    def get_chunks_for_multiple_PDFs(
        self, paths: list, book_names: list = None, chunk_size=3000
    ):
        """
        Get the text chunks for multiple PDF files.
        The paths and book names should be in the same order.
        Args:
            paths (list): The list of paths to the PDF files.
            book_names (list): The list of book names.
            chunk_size (int): The size of the text chunks to split the text into.
        Returns:
            pd.DataFrame: The DataFrame containing the text data with embeddings.
        """
        full_df = pd.DataFrame(columns=["Book", "Chapter", "Text", "Topic"])
        pb = tqdm(enumerate(paths), desc="Processing PDFs", total=len(paths))
        for index, path in pb:
            if book_names is not None:
                book_name = book_names[index]
            else:
                book_name = None
            pb.set_description(f"Processing PDF: {book_name}")

            try:
                df = self.get_chunks_df_from_PDF(path, book_name, chunk_size)
            except Exception as e:
                raise e
            full_df = pd.concat([full_df, df], ignore_index=True)
        return full_df

    def get_embeddings_df_from_multiple_PDFs(
        self, paths: list, book_names: list = None, chunk_size=3000
    ):
        """
        Get the embeddings DataFrame for multiple PDF files.
        The paths and book names should be in the same order.
        Args:
            paths (list): The list of paths to the PDF files.
            book_names (list): The list of book names.
            chunk_size (int): The size of the text chunks to split the text into.
        Returns:
            pd.DataFrame: The DataFrame containing the text data with embeddings.
        """
        full_df = self.get_chunks_for_multiple_PDFs(paths, book_names, chunk_size)
        return self.get_embeddings_df(full_df)

    def set_embeddings_df(self, df: pd.DataFrame):
        """
        Set the embeddings DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame containing the text data with embeddings.
        """
        self.embedding_df = df

    def set_embeddings_df_from_PDF(
        self, path: str, book_name=None, chunk_size=3000, update_index=True
    ):
        """
        Set the embeddings DataFrame from the given PDF file.
        Args:
            path (str): The path to the PDF file.
            book_name (str): The name of the book.
            chunk_size (int): The size of the text chunks to split the text into.
        """
        try:
            df = self.get_chunks_df_from_PDF(path, book_name, chunk_size)
        except Exception as e:
            raise e
        self.set_embeddings_df(self.get_embeddings_df(df))
        if update_index:
            self.create_faiss_index()

    def set_embeddings_df_from_multiple_PDFs(
        self, paths: list, book_names: list = None, chunk_size=3000, update_index=True
    ):
        """
        Set the embeddings DataFrame for multiple PDF files.
        The paths and book names should be in the same order.
        Args:
            paths (list): The list of paths to the PDF files.
            book_names (list): The list of book names.
            chunk_size (int): The size of the text chunks to split the text into.
        """
        full_df = self.get_chunks_for_multiple_PDFs(paths, book_names, chunk_size)
        self.set_embeddings_df(self.get_embeddings_df(full_df))
        if update_index:
            self.create_faiss_index()

    def add_PDF_to_embeddings_df(
        self, path: str, book_name=None, chunk_size=3000, update_index=True
    ):
        """
        Add the text data from the given PDF file to the embeddings DataFrame.
        Args:
            path (str): The path to the PDF file.
            book_name (str): The name of the book.
            chunk_size (int): The size of the text chunks to split the text into.
        """
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )
        if not hasattr(self, "faiss_index") and update_index:
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )

        try:
            df = self.get_chunks_df_from_PDF(path, book_name, chunk_size)
            df_embeddings = self.get_embeddings_df(df)
        except Exception as e:
            raise e
        self.embedding_df = pd.concat(
            [self.embedding_df, df_embeddings], ignore_index=True
        )
        if update_index:
            self.create_faiss_index()

    def add_multiple_PDFs_to_embeddings_df(
        self, paths: list, book_names: list = None, chunk_size=3000, update_index=True
    ):
        """
        Add the text data from multiple PDF files to the embeddings DataFrame.
        The paths and book names should be in the same order.
        Args:
            paths (list): The list of paths to the PDF files.
            book_names (list): The list of book names.
            chunk_size (int): The size of the text chunks to split the text into.
        """
        if not hasattr(self, "embedding_df"):
            raise Exception(
                "Embeddings DataFrame not loaded. Please load the embeddings first."
            )
        if not hasattr(self, "faiss_index") and update_index:
            raise Exception(
                "FAISS index not created. Please create the FAISS index first."
            )
        full_df = self.get_chunks_for_multiple_PDFs(paths, book_names, chunk_size)
        df_embeddings = self.get_embeddings_df(full_df)
        self.embedding_df = pd.concat(
            [self.embedding_df, df_embeddings], ignore_index=True
        )
        if update_index:
            self.create_faiss_index()
