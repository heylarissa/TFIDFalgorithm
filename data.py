import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def read_csv_file(file: str):
    """ Opens csv file in write mode and returns the dataset """
    dataset = pd.read_csv(file)
    file = open("words.txt", "w+")
    return dataset

def remove_special_characteres(text: str):
    words = text.replace("'", "")
    for c in '123456789.,:?/#"':
        words = words.replace(c, "") 

    return words

def remove_stop_words(text: str, stop_words):
    """ Remove stop words from the string """
    doc = ""
    words = text.split(" ")
    for word in words:
        if word not in stop_words:
            doc = f"{doc} {word}"

    return doc

def data_preprocessing(dataset, string_column: str):
    """
    Tokenização: Dividir o texto em palavras ou tokens.
    Remoção de Pontuação e Caracteres Especiais: Eliminar caracteres que não são relevantes para a análise, como pontuações.
    Minúsculas (Lowercasing): Converter todas as letras para minúsculas para garantir que as palavras não sejam tratadas de maneira diferente com base no caso.
    Remoção de Stop Words: Remover palavras comuns (stop words) que não contribuem muito para a análise.
    Stemming ou Lemmatization: Reduzir as palavras à sua forma raiz (stem) ou forma base (lemma) para tratar palavras relacionadas de maneira semelhante.
    Codificação de Rótulos de Classe: Converter as classes em valores numéricos se necessário.

    """

    documents = dataset[string_column]
    cleaned_documents = []
    stop_words = set(stopwords.words("portuguese"))  # palavras não essenciais

    for document in documents:
        doc = remove_special_characteres(document.lower())
        doc = remove_stop_words(doc, stop_words)
        
        # stemming
        filtered_tokens = document.split(" ")
        # stemmer = PorterStemmer()
        # stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
        # for token in stemmed_tokens:
        #     doc = f"{doc} {token}"
        cleaned_documents.append(doc)
        # tf-idf (codificação de rótulos de classe)

    return cleaned_documents
