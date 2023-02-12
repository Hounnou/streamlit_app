import streamlit as st 
import numpy as np 
#import plotly.express as px

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

st.title(':red[Performance of some Machine Learning Models]')

#st.header(':blue[Compare three algorithms]')
st.write("""This project aims to compare the performance of different machine learning models 
            on three datasets: Iris, Breast Cancer and Wine. The machine learning models are:
            Support Vector Machine (SVM), K-Nearest Neighbor(KNN), Random Forest, and Multinomial Naive Bayes.
            Select a model and a dataset to see the performance
            of the model in terms of accuracy.""")

col1, col2 = st.columns([1,2])

dataset_name = st.sidebar.selectbox(
    'Choose a dataset',
    ('Iris', 'Breast Cancer', 'Wine')
)

#col1.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Choose a model',
    ('KNN', 'SVM', 'Random Forest', 'Multinomial Naive Bayes')
)

def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    return X, y

X, y = get_dataset(dataset_name)
#col1.write('Shape of dataset:', X.shape)
#col1.write('number of classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.selectbox('Number of neighbors', (range(1,20)))
        params['K'] = K
    elif clf_name == 'Multinomial Naive Bayes':
        alpha = st.sidebar.slider('alpha', 0.1 , 1.0)
        params['alpha'] = alpha
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'SVM':
        clf = SVC(C=params['C'])
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'Multinomial Naive Bayes':
        clf = MultinomialNB(alpha=params['alpha'])
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)
#### CLASSIFICATION ####

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = 100*round(accuracy_score(y_test, y_pred),1)


col1.write(f"### :blue[{dataset_name} Data]")
col1.write(f"### :blue[{classifier_name} Model]")
col1.metric( label = ":blue[Accuracy:]", value = f"{acc}%")



#### PLOT DATASET ####
# Project the data onto the 2 primary principal components
pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
        c=y, alpha=0.8,
        cmap='viridis')
plt.xlabel('First Component')
plt.ylabel('Second Component')
plt.colorbar()

plt.show()
col2.pyplot(fig)

