from data import read_csv_file, data_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import SVC


def run():
    dataset = read_csv_file("g1_v1.csv")
    documents = data_preprocessing(dataset=dataset, string_column="texto")

    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(documents)

    X_train, X_test, y_train, y_test = train_test_split(
        result, dataset["classe"], test_size=0.3, random_state=42
    )

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42)

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

    svm_model = SVC(kernel="linear", C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)

    y_pred = svm_model.predict(X_test)
    confusion = confusion_matrix(y_test, y_pred)
    print("Matriz de Confusão:")
    print(confusion)
    print()

    accuracy = accuracy_score(y_test, y_pred)
    print("Acurácia: ", accuracy)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    run()
