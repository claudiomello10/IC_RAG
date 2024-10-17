from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
)
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")

Settings.llm = llm

template = (
    "You are an assistant helping a student with their studies in machine learning. \n"
    "You will be provided with different texts from books as context and you should reference in which books and chapters the information can be found so the student can read more about it. \n"
    "The student will ask you questions about the text and you should provide the answer based on the context. \n"
    "Bellow is the context retrieved from the books: \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n\n"
)
qa_template = PromptTemplate(template)


# Load the index
storage_context = StorageContext.from_defaults(persist_dir="./index/llama_index")

index = load_index_from_storage(storage_context, top_k=10)


query_engine = index.as_query_engine(
    text_qa_template=qa_template,
)


for i in range(10):
    input_text = input("User: ")
    response = query_engine.query(input_text)
    print("IC_LLM: ", response)
