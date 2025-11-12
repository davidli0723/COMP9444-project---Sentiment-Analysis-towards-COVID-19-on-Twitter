"""
model constant and hyper tuning model hypertuning params
"""
import sys
import torch
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, get_constant_schedule, AdamWeightDecay,RobertaModel, RobertaTokenizer,AutoTokenizer, AutoModel


# Set device for running model, in case of GPU ,it will automatically use GPU.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
# "vinai/bertweet-base"  "bert-base-cased"
# BERT pre trained model, which defined tokens, buffer etc.
if len(sys.argv) > 1:
    PRE_TRAINED_MODEL = sys.argv[1]
    if PRE_TRAINED_MODEL !="roberta-base":
        tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL)
        bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL)
        bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL)

else:
    print("请传入一个参数")


# Maximum input tensor length based on sentence.
SENTENCE_LENGTH = 128

# Random seed for model
RANDOM_SEED = 35

# Training test size 10%
training_test_size = 0.1

# Validation test size 50%
validation_test_size = 0.5

# Worker for data loader
WORKER = 4

# Batch size
BATCH = 32

# Epoch for training
EPOCHS = 5

# Optimizer parameters for fine tune BERT.
LEARNING_RATE = 1e-5

# Optimizer EPS
EPS = 1e-6

# Correct bias
CORRECT_BIAS = False

# optimizer class
adam = AdamW

# schedule
scheduler = get_linear_schedule_with_warmup

# model save path
SAVE_PATH = 'bin/'

# dataframe path
DATAFRAME = '/content/drive/MyDrive/cleanest.csv'