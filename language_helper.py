import langdetect


def detect(sentence):
    return langdetect.detect(sentence) != 'en'


def clear_nonenglish_words(sentence):
    sentence = sentence.lower();
    english_sentence = ''
    for word in sentence.split():
        if _is_english(word):
            english_sentence = english_sentence+' '+word
    return english_sentence.replace(',','');


def _is_english(s):
    try:
        s.encode('ascii')
    except UnicodeEncodeError:
        return False
    else:
        return True

