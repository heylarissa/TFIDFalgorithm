import pandas as pd
from nltk.corpus import stopwords

"""
Etapas para classificação do texto utilizando o algoritmo TF-IDF

1. Tokenizar as palavras
2. Remover todas as palavras remetidas
3. Somar 1 a cada repetição de uma determinada palavra

"""

def term_frequency():
    pass

dados = pd.read_csv("g1_v1.csv")
file = open("words.txt", "w+")
vocab = []  # palavras distintas
document = []
frequency = []  # frequência de recorrência
stop_words = set(stopwords.words("portuguese"))  # palavras não essenciais
data = []

# merge documents into a single corpus
for text in dados["texto"]:
    # tokenizar (separa em tokens)
    w = text.lower()
    
    # remove números e pontuação
    words = w.replace("'", "")
    for c in '123456789.,:?/#"':
        words = words.replace(c, "") 

    data.append(words)
    
    words = words.split(" ")
    for word in words:
        if word not in stop_words:
            document.append(word)
            if word not in vocab:
                # remove palavras repetidas e stop words
                file.write(word + ", ")
                vocab.append(word)

for word in vocab:
    frequency.append(document.count(word))

words_frequency = []
i = 0
for word in vocab:
    words_frequency.append({'word': word, 'frequency': frequency[i]}) 
    i+=1

print("Tamanho do vocabulário: " + str(len(vocab)))
# print(words_frequency)
file.close()

# import required module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# create object
tfidf = TfidfVectorizer()
 
# get tf-df values
result = tfidf.fit_transform(data)
print(result)
X_train, X_test, y_train, y_test = train_test_split(result, dados['classe'], test_size=0.3, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Divisão dos dados e treinamento do modelo

rf_model = RandomForestClassifier(n_estimators=300, random_state=42) 
# n_estimators = number of trees in the forest

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
confusion = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(confusion)
print()

accuracy = accuracy_score(y_test, y_pred)
print("Acurácia: ", accuracy)
print()
print(classification_report(y_test, y_pred))

