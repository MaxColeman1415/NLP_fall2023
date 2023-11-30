# Documentation at https://github.com/thunlp/OpenPrompt
# Dataset at https://metatext.io/datasets/social-bias-inference-corpus-(sbic)-

# Install the following required packages as needed
# import openprompt
# import sklearn
# import numpy as np
# import scipy
# import spacy
import torch
import torch.nn as nn
import pandas as pd
import math
import regex
import os
import pickle
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tqdm import tqdm
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm, T5TokenizerWrapper
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, r2_score, roc_curve



# Step 1: Define data
base_directory = '/change/to/where/the/unzipped/datasets/are/located/'

# these files should exist before running the program
base_file = base_directory + 'SBIC.v2.dev.csv'
train_file = base_directory + 'SBIC.v2.trn.csv'
test_file = base_directory + 'SBIC.v2.tst.csv'

# these files will be created by the program as needed
checkpoint_file = base_directory + 'checkpoint.pth'
category_0_wordlist_file = base_directory + 'category_0_wordlist.pkl'
category_1_wordlist_file = base_directory + 'category_1_wordlist.pkl'

# hyperparameters and other variables as needed
classes = ["not offensive", "offensive"]
batch_size = 20  # reduce size if too much memory is being used and program crashes
max_seq_length = 32  # reduce size if too much memory is being used
epoch_count = 1  # reduce size for faster training, set to 0 to test an existing model
gradient_accumulation = 5  # increase size for faster training
checkpoint = 5 * gradient_accumulation  # saves a checkpoint after x number of steps
loss_function = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']


# Code module for initializing a base wordlist
def process_dataset(filepath):
    # variables for using the file
    # currently hardcoded for the dataset
    sentences_col = 'post'
    sentence_ID_col = 'HITId'
    isOffensive_col = 'offensiveYN'

    # variables for cleaning data
    alphanumeric_only = r'[^A-Za-z0-9 ]+'
    stopword_list = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # dictionaries
    sentence_list = {}
    category_list = {'0': set(), '1': set()}

    # read in the file
    data = pd.read_csv(filepath)

    for index, row in data.iterrows():
        # read in the current sentence
        sentence = row[sentences_col]
        sentence_ID = row[sentence_ID_col]
        isOffensive = row[isOffensive_col]

        # make sure sentence is unique and that there is a value for offensiveness before processing
        if sentence_ID not in sentence_list and not math.isnan(isOffensive):
            sentence_list[sentence_ID] = sentence

            # process sentence to reduce to alphanumeric, no stop words, stemmed and lemmatized tokens
            alphanum_sentence = regex.sub(alphanumeric_only, '', sentence)
            tokenized_sentence = word_tokenize(alphanum_sentence)

            no_stop_sentence = [token for token in tokenized_sentence if token.lower() not in stopword_list]
            stemlem_sentence = [lemmatizer.lemmatize(stemmer.stem(token)) for token in no_stop_sentence]

            category_list[str(round(isOffensive))].update(stemlem_sentence)

    return category_list['0'], category_list['1']


# check if wordlists have already been created.  If not, create them.
print("checking for wordlists...")
if not os.path.exists(category_0_wordlist_file) or not os.path.exists(category_1_wordlist_file):
    print("building wordlists...")
    category_0_wordlist, category_1_wordlist = process_dataset(base_file)
    with open(category_0_wordlist_file, 'wb') as file:
        pickle.dump(category_0_wordlist, file)
    with open(category_1_wordlist_file, 'wb') as file:
        pickle.dump(category_1_wordlist, file)

# load in wordlists
print("reading wordlists...")
with open(category_0_wordlist_file, 'rb') as file:
    category_0_wordlist = pickle.load(file)
with open(category_1_wordlist_file, 'rb') as file:
    category_1_wordlist = pickle.load(file)

# cleaning the training data - remove duplicates, drop missing codings, round non-binary codings
print("preparing training set...")
trn_data = pd.read_csv(train_file)
trn_data = trn_data.drop_duplicates(subset='post')
trn_data = trn_data.dropna(subset='offensiveYN')
trn_data['offensiveYN'] = trn_data['offensiveYN'].round()

# selects 1000 random samples from the dataset due to computing restrictions
trn_data = trn_data.sample(n=1000, random_state=42)

# build InputExample
trn_InputExample = [
    InputExample(guid=i, text_a=text, label=label)
    for i, (text, label) in enumerate(zip(trn_data['post'], trn_data['offensiveYN']))
]


# cleaning the testing data - remove duplicates, drop missing codings, round non-binary codings
print("preparing testing set...")
tst_data = pd.read_csv(test_file)
tst_data = tst_data.drop_duplicates(subset='post')
tst_data = tst_data.dropna(subset='offensiveYN')
tst_data['offensiveYN'] = tst_data['offensiveYN'].round()

# selects 1000 random samples from the dataset due to computing restrictions
tst_data = tst_data.sample(n=1000, random_state=42)


tst_InputExample = [
    InputExample(guid=i, text_a=text, label=label)
    for i, (text, label) in enumerate(zip(tst_data['post'], tst_data['offensiveYN']))
]
print("Data defined")


# Step 2: Define a Pre-trained Language Model (PLM)
# Can pick any model on huggingface
# Go to https://huggingface.co/docs/transformers/index

plm, tokenizer, model_config, WrapperClass = load_plm("t5", "t5-base")
print("PLM defined")

# Step 3: Define a template
print("building template...")
template = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"}{"mask"}')

print("building dataloaders...")

# Build DataLoaders
trn_dataloader = PromptDataLoader(dataset=trn_InputExample,
                                  template=template,
                                  tokenizer=tokenizer,
                                  tokenizer_wrapper_class=T5TokenizerWrapper,
                                  max_seq_length=max_seq_length,  # may appear as a warning, safe to ignore
                                  decoder_max_length=3,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  teacher_forcing=False,
                                  predict_eos_token=False,
                                  truncate_method="head"
                                  )

tst_dataloader = PromptDataLoader(dataset=tst_InputExample,
                                  template=template,
                                  tokenizer=tokenizer,
                                  tokenizer_wrapper_class=T5TokenizerWrapper,
                                  max_seq_length=max_seq_length,  # may appear as a warning, safe to ignore
                                  decoder_max_length=3,
                                  batch_size=int(batch_size/2),
                                  shuffle=True,
                                  teacher_forcing=False,
                                  predict_eos_token=False,
                                  truncate_method="head"
                                  )
print("Template defined")


# Step 4: Define a Verbalizer
print("Building Verbalizer")
verbalizer = ManualVerbalizer(
    tokenizer=tokenizer,
    classes=classes,
    label_words={
        "not offensive": category_0_wordlist,
        "offensive": category_1_wordlist
    },
)
print("Verbalizer defined")


# Step 5: Combine 2-4 into a PromptModel


print("Initializing model...")
prompt_model = PromptForClassification(
    template=template,
    plm=plm,
    verbalizer=verbalizer,
)

print("checking for existing model...")
if os.path.exists(checkpoint_file):
    print("existing model found!")
    print("Would you like to train the existing model again?")
    user_input = input("0 - No\n"
                       "1 - Yes\n"
                       "Enter selection: ")
    if user_input == 1:
        pretrained = torch.load(checkpoint_file)
        prompt_model.load_state_dict(pretrained['model_state'])

prompt_model = nn.DataParallel(prompt_model)  # an attempt to use parallelism to run faster
print("Model defined")


# Step 6: build optimizer
print("building optimizer...")
optimizer_parameters = [
    {'params': [parameter for named_param,
                parameter in prompt_model.named_parameters()
                if not any(no_dec_flag in named_param for no_dec_flag in no_decay)],
     'weight decay': 0.005},
    {'params': [parameter for named_param,
                parameter in prompt_model.named_parameters()
                if any(no_dec_flag in named_param for no_dec_flag in no_decay)],
     'weight decay': 0.001}
]
optimizer = torch.optim.AdamW(optimizer_parameters, lr=0.001)
print("Ready for training.")


# Step 7: Training
print("*****BEGIN TRAINING*****")
for epoch in range(epoch_count):
    total_loss = 0
    for step, inputs in enumerate(tqdm(trn_dataloader, desc=f'Epoch {epoch + 1}', total=len(trn_dataloader))):
        logits = prompt_model(inputs)
        labels = inputs['label'].long()
        loss = loss_function(logits, labels)
        loss = loss / gradient_accumulation
        loss.backward()

        # torch.nn.utils.clip_grad_norm(prompt_model.parameters(), max_norm=1.0)
        total_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % gradient_accumulation == 0:
            print("Epoch {}, average loss: {}".format(epoch + 1, total_loss / (step + 1)),
                  flush=True)
            optimizer.step()
            optimizer.zero_grad()

            if (step + 1) % checkpoint == 0:
                # saves the current model
                torch.save({
                    'epoch': epoch + 1,
                    'step': step + 1,
                    'model_state': prompt_model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'loss': total_loss / (step + 1),
                }, checkpoint_file)
print("training complete")


# Step 8: Testing
print("*****BEGIN TESTING*****")
predictions = []
all_labels = []
confusion_scores = {'guess_0_actual_0': 0, 'guess_0_actual_1': 0, 'guess_1_actual_0': 0, 'guess_1_actual_1': 0}

for step, inputs in enumerate(tqdm(tst_dataloader, desc='Testing', total=len(tst_dataloader))):
    logits = prompt_model(inputs)
    labels = inputs['label']
    all_labels.extend(labels.cpu().tolist())
    predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    for single_prediction, actual in zip(predictions, all_labels):
        key = f'guess_{int(single_prediction)}_actual_{int(actual)}'
        confusion_scores[key] += 1

# Calculate statistics
accuracy = (sum([int(guess_label == true_label)
                 for guess_label, true_label
                 in zip(predictions, all_labels)])
            / len(predictions))

roc_auc_score = roc_auc_score(all_labels, predictions)
mae = mean_absolute_error(all_labels, predictions)
mse = mean_squared_error(all_labels, predictions)
r2 = r2_score(all_labels, predictions)

# print statistics
print("*****TESTING COMPLETE*****")
print("Accuracy: {:.4%}".format(accuracy))  # want this close to 100%
print("ROC-AUC: {:.4%}".format(roc_auc_score))  # want this close to 100%
print("MAE: {:.4f}".format(mae))  # want this low as possible
print("MSE: {:.4f}".format(mse))  # want this low as possible
print("R2: {:.4f}".format(r2))  # want this close to 1

# print confusion matrix
for key, count in confusion_scores.items():
    print(f"{key}: {count}")

# plot and print the ROC-AUC curve
false_pos, true_pos, _ = roc_curve(all_labels, predictions)
plt.figure()
plt.plot(false_pos, true_pos, color='black', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.title('ROC-AUC Curve')
plt.legend(loc='lower right')
plt.savefig(base_directory + 'roc_curve.png')


# Citations
# @article{ding2021openprompt,
#  title={OpenPrompt: An Open-source Framework for Prompt-learning},
#  author={Ding, Ning and Hu, Shengding and Zhao, Weilin and Chen,
#  Yulin and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong},
#  journal={arXiv preprint arXiv:2111.01998},
#  year={2021}
# }
