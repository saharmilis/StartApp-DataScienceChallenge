import numpy as np
import pandas as pd
from sklearn import model_selection

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pickle


df = pd.DataFrame.from_csv('train.csv');

print(df.shape)
print()
array_values = df.values
input = array_values[:,[0,1]]
output = array_values[:,2]

###
# vectorizer = CountVectorizer(min_df=1,stop_words='english');
vectorizer = TfidfVectorizer(min_df=1,stop_words='english');
descriptions = array_values[:,1]
bag_of_words = vectorizer.fit_transform(descriptions)
# vectorizer.vocabulary_.get("you");
# print(bag_of_words)

# print(vectorizer)
# print(len(vectorizer.get_feature_names()))
# print(bag_of_words.toarray())

###


validation_size = 0.2
seed = 7
scoring = 'accuracy'

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(bag_of_words.toarray(),output, test_size=validation_size, random_state=seed,shuffle=True)

print(X_train.shape)
print(Y_train.shape)
print()
print(X_validation.shape)
print(Y_validation.shape)

# print('-----------------------------')
print('-----------------------------')
#
# # Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

print("-----------------------------------")



# model = KNeighborsClassifier()
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)
# print(accuracy_score(Y_validation, predictions))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))

# model = KNeighborsClassifier()
# # model = GaussianNB()
# kfold = model_selection.KFold(n_splits=10, random_state=seed)
# cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
# msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())
# print(msg)


# # save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))
#
# # load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# predictions = loaded_model.predict(X_validation)
# # result = loaded_model.score(X_validation, Y_validation)
# print(accuracy_score(Y_validation, predictions))
# # print(result)

# from sklearn.externals import joblib
# from sklearn.datasets import load_digits
# from sklearn.linear_model import SGDClassifier

# filename = 'digits_classifier.joblib.pkl'
# save the classifier
# _ = joblib.dump(model, filename, compress=9)
# load it again
# clf2 = joblib.load(filename)
# print(clf2.score(X_validation, Y_validation))

