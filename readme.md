# IC (Inteligência Computacional) RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed to help students study machine learning concepts using content from popular machine learning textbooks. The system features a modern chat interface and uses advanced language models to provide context-aware responses.

## Features

- **Interactive Chat Interface**: Modern UI built with CustomTkinter
- **Retrieval-Augmented Generation**: Leverages textbook content to provide accurate, contextual responses
- **Conversation Mode**: Supports both standalone queries and continuous conversations
- **Theme Customization**: System, Light, and Dark mode support
- **Efficient Document Processing**: Automated chunking and embedding generation for textbook content

## Quick Start (Using Pre-processed Embeddings)

If you want to quickly run the project without processing the books yourself:

1. Clone the repository:

```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:

```bash
# Note: faiss-gpu is available but not recommended for this project
# It takes longer to compile and the performance difference is minimal for queries
# Only consider faiss-gpu if you're processing new books with very large datasets

pip install -r requirements.txt
```

4. Set up your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'  # On Unix/MacOS
set OPENAI_API_KEY='your-api-key-here'     # On Windows
```

5. Run the application:

```bash
python UI_app_v0.1.0.py
```

The repository includes pre-processed embeddings files (`full_df_embeddings_sections1.json` and `full_df_embeddings_sections2.json`), so you can start using the system immediately.

## Full Setup (Processing Books Yourself)

If you want to process the books yourself or add new books to the system, follow these additional steps:

### 1. Download and Process Books

1. Download the machine learning textbooks:

   - Access the books from: [ML Books Google Drive](https://drive.google.com/file/d/1_mYgjK9-vnYD68pHgPie9K8FmYaB9Lzy/view?usp=sharing)
   - Extract the downloaded file
   - Place the books in a directory named `Livros/All` in your project folder
2. Process the books:

```bash
python index_all_books.py
```

3. Generate embeddings:

```bash
python generate_embeddings_from_csv.py
```

This will create new embedding files that will replace the existing ones.

## Required Dependencies

- PyMuPDF (fitz)
- pandas
- tqdm
- PyPDF2
- langchain
- sentence-transformers
- torch
- faiss-cpu (recommended) or faiss-gpu (optional, longer compilation time)
- customtkinter
- openai
- nltk
- ujson

### Note on FAISS Installation

- **Recommended**: Use `faiss-cpu` for this project

  - Quick to install
  - Sufficient performance for querying the existing embeddings
  - No compilation required
- **Optional**: Use `faiss-gpu` only if:

  - You're processing a very large number of new books
  - You have a compatible GPU
  - You're willing to wait for the longer compilation time

## Project Structure

```
.
├── UI_app_v0.1.0.py                    # Main application with GUI
├── LLM_IC.py                          # Core RAG implementation
├── Set_Chapters_Ui.py                 # Document processing interface
├── index_all_books.py                 # Book indexing script
├── generate_embeddings_from_csv.py    # Embedding generation script
├── full_df_embeddings_sections1.json  # Pre-processed embeddings part 1
├── full_df_embeddings_sections2.json  # Pre-processed embeddings part 2
└── Livros/                           # Optional - needed only if processing books
    └── All/                          # Directory containing the textbooks
```

## Features in the Chat Interface:

- **Conversation Mode**: Toggle between single-query and continuous conversation modes
- **Theme Selection**: Choose between System, Light, and Dark themes
- **Clear Chat**: Reset the conversation history
- **Send Messages**: Use the text input field or press Enter to send queries

## Notes on Usage

1. The system uses pre-processed embeddings by default
2. For optimal performance with minimal setup overhead, use faiss-cpu
3. The embeddings are split into two JSON files for efficient memory management
4. The conversation mode maintains context across multiple queries
5. You only need to process the books yourself if you want to:
   - Add new books to the system
   - Modify the chunking parameters
   - Create custom embeddings

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Built using the jinaai/jina-embeddings-v3 model for embeddings
- Uses OpenAI's language models for response generation
- CustomTkinter for the modern UI implementation
- Machine Learning textbooks provided for educational purposes
