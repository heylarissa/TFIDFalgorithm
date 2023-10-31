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

    X_train, X_test, y_train, y_test = train_test_split(
        documents, dataset["classe"], test_size=0.2, random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # # Calcule os valores de TF-IDF
    # tfidf_values = X_train_tfidf.max(0).toarray()[0]

    # # Use argsort para obter os índices das palavras em ordem decrescente de TF-IDF
    # top_n = 300  # Escolha o número desejado de palavras com os maiores valores de TF-IDF
    # top_indices = np.argsort(-tfidf_values)[:top_n]
    # # Obtenha os nomes das palavras correspondentes
    # feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
    # selected_feature_names = feature_names[top_indices]

    # # Redefina o vetorizador TF-IDF para usar apenas as palavras selecionadas
    # tfidf_vectorizer_selected = TfidfVectorizer(vocabulary=selected_feature_names)

    # # Transforme os dados de treinamento e teste para usar apenas as palavras selecionadas
    # X_train_tfidf_selected = tfidf_vectorizer_selected.fit_transform(X_train)
    # X_test_tfidf_selected = tfidf_vectorizer_selected.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_tfidf, y_train)

    svm_model = SVC(kernel="linear", C=1, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)


    rf_predictions = rf_model.predict(X_test_tfidf)
    svm_predictions = svm_model.predict(X_test_tfidf)

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    svm_accuracy = accuracy_score(y_test, svm_predictions)

    # Imprima os resultados
    print("Desempenho do Random Forest:")
    print(f"Acurácia: {rf_accuracy:.2f}")
    print(classification_report(y_test, rf_predictions))
    confusion = confusion_matrix(y_test, rf_predictions)
    print(confusion)

    print("\nDesempenho do SVM:")
    print(f"Acurácia: {svm_accuracy:.2f}")
    print(classification_report(y_test, svm_predictions))
    confusion = confusion_matrix(y_test, svm_predictions)
    print(confusion)

if __name__ == "__main__":
    run()
