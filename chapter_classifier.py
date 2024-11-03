import fitz  # PyMuPDF
from transformers import pipeline

# Step 1: Choose a PDF file

#PATH = "Livros/Futuros/MachineLearning/2024_Python_Deep_Learning_w_pacc21.pdf" 
#PATH = "Livros/Futuros/MachineLearning/2019_Book_EmbeddedDeepLearning.pdf"
#PATH = "Livros/Futuros/MachineLearning/2018_Book_DeepLearningWithAzure.pdf"
PATH = "Livros/Futuros/MachineLearning/2018_Book_DeepLearningWithApplicationsUsingPython.pdf"
#PATH = "Livros/Futuros/MachineLearning/2017_Book_DeepLearningWithPython.pdf" 
#PATH = "Livros/Futuros/MachineLearning/2017_Book_ProDeepLearningWithTensorFlow.pdf" 
#PATH = "/home/rebecca/IC_RAG/Livros/Futuros/MachineLearning/Adrian Rosebrock-Deep Learning for Computer Vision with Python. 2-Practitioner Bundle-PyImageSearch (2017).pdf"
#PATH = "Livros/Futuros/MachineLearning/2022_Aurélien Géron - Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ Concepts, Tools, and Techniques to Build Intelligent Systems-O'Reilly Media (2022).pdf"
#PATH = "Livros/Futuros/MachineLearning/François Chollet-Deep Learning with Python-Manning (2018).pdf"
#PATH = "Livros/Futuros/MachineLearning/Sebastian Raschka-Python Machine Learning-Packt Publishing (2015).pdf"
#PATH = "Livros/Futuros/MachineLearning/daumé-imitation-learning.pdf"
#PATH = "Livros/Futuros/MachineLearning/2015_machine_learning_a_bayesian_perspective.pdf"
#PATH = "Livros/Futuros/MachineLearning/2018_Aggarwal_NeuralNetworksAndDeepLearning.pdf"
#PATH = "Livros/Futuros/MachineLearning/2022_Chuo_Convex Optimization for Machine Learning.pdf"
#PATH = "Livros/Futuros/MachineLearning/Adrian Rosebrock-Deep Learning for Computer Vision with Python. 2-Practitioner Bundle-PyImageSearch (2017).pdf"
#PATH = "Livros/Futuros/MachineLearning/Adrian Rosebrock-Deep Learning for Computer Vision with Python. 3-ImageNetBundle-PyImageSearch (2017).pdf"
#PATH = "Livros/Futuros/MachineLearning/Adrian Rosebrock-Deep Learning for Computer Vision with Python. 1,Starter Bundle-PyImageSearch (2017).pdf"

# Step 2: Open the PDF file
doc = fitz.open(PATH)

# Step 3: Extract the bookmarks (table of contents) and divide into "main section" and "subsection"
toc = doc.get_toc()

main_section = []
sub_section = []

# Step 5: Print the bookmarks
for i in range(len(toc)):
    if(toc[i][0] == 1 ):
        main_section.append(toc[i]) 
        print(toc[i])   
    if(toc[i][0] == 2 ):
        sub_section.append(toc[i]) 

candidate_labels1 = [ "Main chapter", "Appendix", "Index",  "Acronyms Section", "Preface", "Acknowledgments",  "References" ]
candidate_labels2 = [ "Main chapter", "index", "references" , "summary"]

hypothesis_template = "This text title is a {} section of the book."


zeroshot_classifier = pipeline("zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
first = []

for title in main_section:
    strings = [item for item in title if isinstance(item, str)]
    output = zeroshot_classifier(strings, candidate_labels1, hypothesis_template=hypothesis_template, multi_label=False)
    output = output[0]
    for label, score in zip(output['labels'], output['scores']):
        if label == "Main chapter" and score > 0.30: 
            print(f' Main chapter: {strings},  Class: {label}, Score: {score:.2f}')  


print("--------subsection---------------")
for title in sub_section:
    strings = [item for item in title if isinstance(item, str)]
    output = zeroshot_classifier(strings, candidate_labels2, hypothesis_template=hypothesis_template, multi_label=False)
    output = output[0]
    for label, score in zip(output['labels'], output['scores']):
        if label == "Main chapter" and score > 0.20: 
            print(f' Subtitle: {strings},  Class: {label}, Score: {score:.2f}')  


