import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    processed_text = ' '.join(tokens)
    return processed_text

def load_data(filepath):
    try:
        data = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip', engine='python')
    except pd.errors.ParserError:
        data = pd.read_csv(filepath, delimiter=';', on_bad_lines='skip', engine='python')
    return data

train_data = load_data('ReutersGrain-train.csv')
test_data = load_data('ReutersGrain-test.csv')

# Verificando as colunas e renomeando se necessário
print("Train Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

if 'Text' in train_data.columns and 'class-att' in train_data.columns:
    train_data = train_data.rename(columns={'Text': 'text', 'class-att': 'label'})
if 'Text' in test_data.columns and 'class-att' in test_data.columns:
    test_data = test_data.rename(columns={'Text': 'text', 'class-att': 'label'})

print(train_data.head())
print(test_data.head())

# Aplicando pré-processamento
if 'text' in train_data.columns and 'text' in test_data.columns:
    train_data['processed_text'] = train_data['text'].apply(preprocess)
    test_data['processed_text'] = test_data['text'].apply(preprocess)
else:
    raise KeyError("A coluna 'text' não está presente nos dados de treinamento ou teste.")

# Vetorização
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])

y_train = train_data['label']
y_test = test_data['label']

# Treinando modelos
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Avaliando modelos
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)

print(f'Accuracy of Naive Bayes: {nb_accuracy}')
print(f'Accuracy of SVM: {svm_accuracy}')
