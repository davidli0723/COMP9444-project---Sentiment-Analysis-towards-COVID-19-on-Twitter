"""
data preprocessor step.
1) taking input raw text
2) tokenize it. i.e adding special BERT classification token and buffer tokens
3) convert text into special BERT requirements , input_ids and attention_mask
4) returning object that has preprocessed data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from config.parameters import SENTENCE_LENGTH, BATCH, WORKER
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# def preprocess_text(text):
#     # 1. 去除噪声：移除URL、标签等特殊字符
#     text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
#     text = re.sub(r'<.*?>', '', text)

#     # 2. 统一为小写
#     text = text.lower()

#     # 3. 分词与词形还原
#     words = word_tokenize(text)
#     words = [lemmatizer.lemmatize(word) for word in words if word.isalpha()]

#     # 4. 去除停用词
#     words = [word for word in words if word not in stop_words]

#     return ' '.join(words)

class PreProcessor(Dataset):
    """
    pre-processing text, returning object with attention mask and encoded text
    """



    def __init__(self, text, sentiment, tokenizer):
        self.text = text
        self.sentiment = sentiment
        self.tokenizer = tokenizer

    def __str__(self):
        return ""

    def __len__(self):
        if self.text is not None:
            return len(self.text)

    def __getitem__(self, item):
        if self.text[item] is not None:
            encoded_string = self.tokenizer.encode_plus(self.text[item],
                                                         max_length=SENTENCE_LENGTH,
                                                         return_attention_mask=True,
                                                         pad_to_max_length=True,
                                                         add_special_tokens=True,
                                                         return_tensors='pt')
            preprocessed = {
                "text": self.text[item],
                "sentiment": torch.tensor(self.sentiment[item], dtype=torch.long),
                "input_ids": encoded_string["input_ids"].flatten(),
                "attention_mask": encoded_string["attention_mask"].flatten()
            }

            return preprocessed
        else:
            print("*"*100)
            print("some text has issue")
            print("*"*100)


class GetLoader:
    """
    data loader class
    """
    def __init__(self, dataframe, tokenizer):
        self.df = dataframe
        self.tokenizer = tokenizer

    def get(self):
        return DataLoader(PreProcessor(text=self.df.processed.to_numpy(),
                                       sentiment=self.df.label.to_numpy(),
                                       tokenizer=self.tokenizer,
                                       ),
                          batch_size=BATCH,
                          num_workers=WORKER)





# df = pd.read_csv('../data/data.csv')
# print(df.head())
#
# tl = GetLoader(df, tokenizer)
# train_loader = tl.get()