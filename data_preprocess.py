import language_helper

# clear unused columns
# clear non english apps
# clear all non english words
# clear all stopwords english words - not a must


def clear_colums():
    pass


def clear_non_english_apps():
    pass


def clear_all_non_enghlish_words():
    pass


def clear_all_stopwords():
    pass


def predict_all_data_requested():
    import pandas as pd
    from pandas import ExcelWriter
    from model import model_predict
    from server import prediction

    # from excel to DF
    file_name = 'appDescriptions2.xlsx';
    xl = pd.ExcelFile(file_name)
    df = xl.parse("Classify")


    print(df.shape)
    print(df.values[:,3]) # segment
    print(df.values[:,4]) # description

    for x in range(1100):
        pre = prediction(df.values[x,4]);
        print(pre)
        df.set_value(x,3,pre)
        # print(df.iloc[x,'segment'])

    # from DF to excel
    writer = ExcelWriter('PythonExport.xlsx')
    df.to_excel(writer, 'Sheet5')
    writer.save()



predict_all_data_requested()