import pandas as pd
from nltk.corpus import stopwords

"""
Etapas para classificação do texto utilizando o algoritmo TF-IDF

1. Tokenizar as palavras
2. Remover todas as palavras remetidas
3. Somar 1 a cada repetição de uma determinada palavra

"""

dados = pd.read_csv('g1_v1.csv')
file = open('words.txt', 'w+')
document = []
stop_words = set(stopwords.words('portuguese'))

for text in dados['texto']:
    # tokenizar (separa em tokens)
    words = text.lower().split(' ')
    
    for word in words:

        # remove números e pontuação
        w = word.replace("'", '')

        for c in '123456789.,:?/#"':
            w = w.replace(c, '')

        if w not in document and w not in stop_words:
            file.write(w + ', ')
            document.append(w)

print('Tamanho do vocabulário: ' + str(len(document)))

file.close()

