from MultiPurpuse_LLM import StudentHelper

# Create the LLM_IC object
llm = StudentHelper()

llm.set_embeddings_df_from_multiple_PDFs(
    [
        "Livros/All/2-Aurelien-Geron-Hands-On-Machine-Learning-with-Scikit-Learn-Keras-and-Tensorflow_-Concepts-Tools-and-Techniques-to-Build-Intelligent-Systems-OReilly-Media-2019.pdf"
    ]
)
