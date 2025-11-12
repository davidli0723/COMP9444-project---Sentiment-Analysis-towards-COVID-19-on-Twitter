import re
import nltk
from nltk.corpus import stopwords, words, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import emoji
import string

# Make sure you have downloaded the resources for NLTK
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("words")
# nltk.download("wordnet")
# nltk.download("omw-1.4")
# nltk.download("averaged_perceptron_tagger_eng")

# If it shows that nltk was not found, check and write the location
nltk.data.path.append('C:\\Users\\50643\\AppData\\Roaming\\nltk_data')


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()

    lemmatized_sentence = []
    for word, tag in tagged:
        wordnet_pos = get_wordnet_pos(tag) or wordnet.NOUN
        lemmatized_word = lemmatizer.lemmatize(word, pos=wordnet_pos)
        lemmatized_sentence.append(lemmatized_word)

    return ' '.join(lemmatized_sentence)


def strip_emoji(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols and pictographs
        "\U0001F680-\U0001F6FF"  # Transport and map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric shapes
        "\U0001F800-\U0001F8FF"  # Supplemental arrows
        "\U0001F900-\U0001F9FF"  # Clothing and objects
        "\U0001FA00-\U0001FA6F"  # Tools and equipment
        "\U0001FA70-\U0001FAFF"  # Other symbols and pictographs
        "\U00002702-\U000027B0"  # Miscellaneous symbols
        "\U000024C2-\U0001F251"  # Enclosed characters and additional pictographs
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


#clean hashtags at the end of the sentence, and keep those in the middle of the sentence by removing just the # symbol
def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in
                         re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet))  #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in
                          re.split('#|_', new_tweet))  #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2


def preprocess_text(text, config_p={}):

    lemma = config_p.get("lemma", 1)
    stem = config_p.get("stem", 1)
    r_number = config_p.get("r_number", 0)
    r_emoji = config_p.get("r_emoji", 1)
    hashtag = config_p.get("hashtag", 0)
    r_repeated_chars = config_p.get("r_repeated_chars", 0)
    r_special_chars = config_p.get("r_special_chars", 0)
    eng_only = config_p.get("eng_only", 0)

    if r_emoji:
        text = strip_emoji(text)

    if hashtag:
        text = re.sub(r'#\S+', '', text)
    else:
        text = text.replace('#', '')

    if r_number:
        text = re.sub(r'\d+', '', text)

    text = text.replace('\r', '').replace('\n', ' ').replace('\n', ' ').lower()  # remove \n and \r and lowercase

    # remove http links and mentions
    result = re.sub(r"(@[\w]+|https?://\S+)", '', text)
    text = re.sub(r'[^\x00-\x7f]', r'', result)  #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'

    # remove http links and mentions
    if r_repeated_chars:
        text = re.sub(r'(.)\1+', r'\1\1', text)

    # remove special characters
    if r_special_chars:
        banned_list = string.punctuation + ''.join(chr(i) for i in range(128, 256))
        table = str.maketrans('', '', banned_list)
        text = text.translate(table)

    # lemmatize
    if lemma:
        text = lemmatize_sentence(text)

    if eng_only:
        text = re.sub(r'[^a-z\s]', '', text)

    word_l = text.split()

    # Stemmer
    if stem:
        stemmer = PorterStemmer()
        word_l = [stemmer.stem(w) for w in word_l]

    stop_words = set(stopwords.words('english'))
    word_l = [word for word in word_l if word not in stop_words and len(word) > 2]

    processed_t = ' '.join(word_l)

    return processed_t


def preprocessing_dataset(source='data/COVIDSenti-B.csv', target='data/COVIDSenti_B_base.csv', split_dataset=1, config_p={}):
    df_a = pd.read_csv(source, encoding='utf-8')

    peek = df_a[:int(split_dataset * len(df_a))].copy()

    import time
    t0 = time.time()
    peek['processed'] = [preprocess_text(text=i, config_p=config_p) for i in peek.tweet]
    t1 = time.time()
    print('time cost:', t1 - t0)

    peek.to_csv(target, encoding='utf-8')


if __name__ == '__main__':

    ''' If the variable is not written in the config dictionary, the default value only set lemma to 1 and r_emoji to 1
    :param lemma: 1 means lemmatize is enabled
    :param stem: 1 means stemming is enabled, you can use lemmatize and stemming at the same time, but it's not recommended. 
    :param r_number: 1 means that all numbers are forcibly deleted.
    :param r_emoji: 1 means retain the content after the hashtag.
    :param hashtag: 1 means that the content after the hashtag is deleted. The default value 0 means that only the hashtag is deleted
    :param r_repeated_chars: 1 means spelling correction and repetitive letter processing.  "coooool" -> "cool" (maybe not needed in models based on twitter)
    :param r_special_chars: including ! " # $ % & \ ' ( ) * + , - . / : ; < = > ? @ [ \ \ ] ^ _ ` { | } ~ Ã ± ã ¼ â » §
    :param eng_only: 1 means only Engilsh which means all other processing will be overwritten. (not recommended) 
    '''

    config = {
        "lemma": 1,
        "stem": 0,
        "r_number": 0,
        "r_emoji": 1,
        "hashtag": 0,
        "r_repeated_chars": 0,
        "r_special_chars": 0,
        "eng_only": 0
    }

    preprocessing_dataset(source='data/COVIDSenti-B.csv', target='test_dataset/B_base_lemmatize.csv', split_dataset=0.2, config_p=config)


