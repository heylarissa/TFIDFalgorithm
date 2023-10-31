from data import read_csv_file, data_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC
import numpy as np

def run():
    dataset = read_csv_file("g1_v1.csv")
    documents = data_preprocessing(dataset=dataset, string_column="texto")

    # Divida os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(documents, dataset["classe"], test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Crie e treine o modelo Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)

    # Crie e treine o modelo SVM
    svm_model = SVC(kernel='linear', C=1, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)

    # Faça previsões com ambos os modelos
    rf_predictions = rf_model.predict(X_test_tfidf)
    svm_predictions = svm_model.predict(X_test_tfidf)

    # Avalie o desempenho dos modelos
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Imprima os resultados
    print("Desempenho do Random Forest:")
    print(f"Acurácia: {rf_accuracy:.2f}")
    print(classification_report(y_test, rf_predictions))

    print("\nDesempenho do SVM:")
    print(f"Acurácia: {svm_accuracy:.2f}")
    print(classification_report(y_test, svm_predictions))
    # # Pré-processamento dos dados de texto usando TF-IDF
    # tfidf_vectorizer = TfidfVectorizer()
    # X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # # top_300_indices = np.argsort(result.max(axis=0).toarray()[0])[::-1][:4000]
    # # filtered_tfidf = result[:, top_300_indices]

    # rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf_model.fit(X_train_tfidf, y_train)
    # rf_predictions = rf_model.predict(X_test_tfidf)
    # confusion = confusion_matrix(y_test, rf_predictions)
    
    # print("Random Forest")
    # print("Matriz de Confusão:")
    # print(confusion)
    # print()

    # rf_accuracy = accuracy_score(y_test, rf_predictions)
    # print("Acurácia: ", rf_accuracy)
    # print()
    # print(classification_report(y_test, rf_predictions))

    # svm_model = SVC(kernel='linear', C=1, random_state=42)
    # svm_model.fit(X_train_tfidf, y_train)


    # svm_predictions = svm_model.predict(X_train_tfidf)
    # confusion = confusion_matrix(y_test, svm_predictions)
    
    # print("SVM")
    # print("Matriz de Confusão:")
    # print(confusion)
    # print()

    # svc_accuracy = accuracy_score(y_test, svm_predictions)
    # print("Acurácia: ", svc_accuracy)

    # print(classification_report(y_test, svm_predictions))


if __name__ == "__main__":
    run()
