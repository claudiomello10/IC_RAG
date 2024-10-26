import fitz  # PyMuPDF
from transformers import pipeline
# Step 2: Define the path to the PDF file
#PATH = "Livros/Futuros/MachineLearning/2024_Python_Deep_Learning_w_pacc21.pdf" # -- bom
#PATH = "Livros/Futuros/MachineLearning/2019_Book_EmbeddedDeepLearning.pdf"
#PATH = "Livros/Futuros/MachineLearning/2018_Book_DeepLearningWithAzure.pdf"
#PATH = "Livros/Futuros/MachineLearning/2018_Book_DeepLearningWithApplicationsUsingPython.pdf"
#PATH = "Livros/Futuros/MachineLearning/2017_Book_DeepLearningWithPython.pdf"
PATH = "Livros/Futuros/MachineLearning/2017_Book_ProDeepLearningWithTensorFlow.pdf"
#PATH = "Livros/Futuros/MachineLearning/'2022_Aurélien Géron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ Concepts, Tools, and Techniques to Build Intelligent Systems-O'\''Reilly Media (2022).pdf'"
#PATH = "Livros/Futuros/MachineLearning/Adrian Rosebrock-Deep Learning for Computer Vision with Python. 3-ImageNetBundle-PyImageSearch (2017).pdf'"

# Step 3: Open the PDF file
doc = fitz.open(PATH)

# Step 4: Extract the bookmarks (table of contents)
toc = doc.get_toc()

section = []
# Step 5: Print the bookmarks
for i in range(len(toc)):
    if(toc[i][0] == 1):
        section.append(toc[i]) 
        print(toc[i])   


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["Key Chapter Content", "appendix", "Index", "Acronyms" "Preface", "Acknowledgments", "A topic related to the Key Chapter Content", "Preliminary Section"]

hypothesis_template = "This title represents which type of section in the book? {}"

'''
for item in section:
    sequence_to_classify = f"Title: '{item[1]}'"
    
    result = classifier(sequence_to_classify, candidate_labels)
    
    if result['labels'][0] == "Key Chapter Content" and result['scores'][0] > 0.20:
        print(f"Key Chapter Content: {item[1]} (page {item[2]})  score: {result['scores'][0]:.2f}")
'''

zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")


for title in section:
    strings = [item for item in title if isinstance(item, str)]
    output = zeroshot_classifier(strings, candidate_labels, hypothesis_template=hypothesis_template, multi_label=False)
    output = output[0]
    for label, score in zip(output['labels'], output['scores']):
        if label == "Key Chapter Content" and score > 0.20: 
            print(f' Título: {strings},  Classe: {label}, Score: {score:.2f}')  

