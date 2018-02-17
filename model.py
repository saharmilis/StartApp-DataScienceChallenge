import numpy as np
import pandas as pd
from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.externals import joblib
import os.path
import time


file_name_data_train = 'train.csv'
file_name_model_trained = 'model_trained.joblib.pkl'
file_name_model_vectorizer = 'model_vectorizer.joblib.pkl'
model = KNeighborsClassifier()
vectorizer = TfidfVectorizer(stop_words='english');


def model_initialize():
    print('model_initialize')
    if os.path.isfile(file_name_model_trained):
        # model exists - load
        try:
            _model_load()
            return
        except:
            pass

    # model no exists - train
    _model_train()


def model_predict(description):
    global model
    global vectorizer
    x_vector = vectorizer.transform([description]);
    return model.predict(x_vector.toarray());


def model_insert_new_data(appId,segment,description):
    # it's not idle
    # did not find how to feed new data to the classifier!! tried StackOverFlow. will update on that.
    indexing_lines = int(time.time())
    f = open(file_name_data_train, 'a')
    f.write('\n'+str(indexing_lines)+','+str(appId)+','+str(segment)+','+str(description))  # python will convert \n to os.linesep
    f.close()  # you can omit in most cases as the destructor will call it
    # train
    _model_train()



def _model_train():
    print('model_train')
    global model
    global vectorizer
    global indexing_lines
    global segments

    # get dataframe
    df = pd.DataFrame.from_csv(file_name_data_train);

    # split to X&Y
    array_values = df.values
    descriptions = array_values[:, 1]
    output = array_values[:,2]

    # from strings to numbers & normalize
    bag_of_words = vectorizer.fit_transform(descriptions)

    # set training parameters
    validation_size = 0 # 0.2 for validation. 0 in production
    seed = 7

    # split data
    x_train, x_validation, y_train, y_validation = model_selection.train_test_split(bag_of_words.toarray(),output, test_size=validation_size, random_state=seed,shuffle=True)

    # set classifier
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    # predictions = model.predict(x_validation)
    # print(predictions)

    # save model
    _model_save()


def _model_load():
    # load the model instead of traning
    global model
    global vectorizer
    model = joblib.load(file_name_model_trained)
    vectorizer = joblib.load(file_name_model_vectorizer)

def _model_save():
    # save the model after traning
    global model
    _ = joblib.dump(model, file_name_model_trained, compress=9)
    _ = joblib.dump(vectorizer, file_name_model_vectorizer, compress=9)
    pass


def get_segments():
    df = pd.DataFrame.from_csv(file_name_data_train);
    segments = df.segment.unique()
    print(segments)
    return segments;

def get_n_neighbors(description, n):
    global model
    global vectorizer

    # get a representation of the app as vector
    x_vector = vectorizer.transform([description]);

    # get the n NN to the vector - return sorted by 'closeness'
    results = model.kneighbors(X=x_vector.toarray(),n_neighbors=n,return_distance=True)
    df = pd.DataFrame.from_csv(file_name_data_train);

    # translate the results to the segments
    results_segments = df.values[results[1][0], 2]

    # Relative Score - scores indicate how 'close' is a segment relative to the closest segment
    results_score = [x / results[0][0][0] for x in results[0][0]]


    return zip(results_segments,results_score)



model_initialize();
if __name__ == '__main__':
    print('main')
    model_initialize();

