import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('vader_lexicon')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))


def preprocess_text(text):
    
    text = text.lower()
    # Remove caracteres especiais, números e pontuações
    text = re.sub(r'[^a-záàâãéèêíïóôõöúçñ\s]', '', text)
    # Tokenização
    words = [word for word in text.split() if word not in stop_words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    

    return ' '.join(lemmatized_words)

def count_of_words(df):
    count_vectorizer = CountVectorizer(max_features=5000)
    
    count_matrix = count_vectorizer.fit_transform(df['processed_review'])
    
    return count_matrix, count_vectorizer

def tfidf_vectorization(df):

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_review'])

    return tfidf_matrix, tfidf_vectorizer


    

def main():
    df = pd.read_csv('olist_order_reviews_dataset.csv')
    

    #Combina o titulo e a mensagem da avaliação em uma única coluna
    df['full_review'] = df['review_comment_title'].fillna('') + ' ' + df['review_comment_message'].fillna('')

    #Remove avaliações nulas
    df = df[df['full_review'].str.strip() != '']
    df['processed_review'] = df['full_review'].apply(preprocess_text)


    #Criação de rótulos baseados em <= 3 negativo e > 3 positivo
    df = df[df['review_score'].isin([1, 2, 4, 5])].copy()
    df['sentiment_label'] = np.where(df['review_score'] > 3, 1, 0)

    #Representação numérica usando TF-IDF, problema: esparsidade na matriz por conta
    # de poucas palavras por avaliação
    tfidf_matrix, tfidf_vectorizer = tfidf_vectorization(df)


    

    X = tfidf_matrix
    y = df['sentiment_label']

    # Divisão em conjunto de treino e teste para calculo de métricas
    # usando estratificação para manter a proporção das classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    regressao = LogisticRegression(random_state=42)
    regressao.fit(X_train, y_train)
    y_pred = regressao.predict(X_test)

    # Desempenho do modelo
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação usando tf-idf:\n", classification_report(y_test, y_pred))

    X, count_vectorizer = count_of_words(df)
    y = df['sentiment_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    regressao = LogisticRegression(random_state=42)
    regressao.fit(X_train, y_train)
    y_pred = regressao.predict(X_test)

    # Desempenho do modelo
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("\nRelatório de Classificação usando contagem de palavras:\n", classification_report(y_test, y_pred))

    feature_names = tfidf_vectorizer.get_feature_names_out()
    coefficients = regressao.coef_[0]
    df_coefficients = pd.DataFrame({'word': feature_names, 'coefficient': coefficients})

    # Ordena as palavras pelo coeficiente de forma decrescente
    # Valores altos indicam palavras mais associadas a avaliações positivas, então elas serão
    # as primeiras, se colocarmos em ordem decrescente
    df_coefficients = df_coefficients.sort_values(by='coefficient', ascending=False)

    print("--- Principais Palavras Positivas ---")
    print(df_coefficients.head(20))

    print("\n--- Principais Palavras Negativas ---")
    # As palavras negativas são as de menor coeficiente (mais negativas)
    print(df_coefficients.tail(20))

if __name__ == "__main__":
    main()