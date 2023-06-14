import os

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import re
import copy
from tqdm.notebook import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from transformers import (
    T5Tokenizer,
    T5Model,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    MT5ForConditionalGeneration, AutoTokenizer
)

# import keras
# %load_ext autoreload
# %autoreload 2
# sns.set_theme()


os.environ["GCLOUD_PROJECT"] = "lego-prod"
tqdm.pandas()

SAVE_DF = False




df = pd.read_csv("1.csv", sep='\t')
df.tags = df.tags.apply(lambda x: ast.literal_eval(x))
df.texts = df.texts.apply(lambda x: ast.literal_eval(x))
df['texts_join'] = df.texts.apply(lambda x: ' '. join(x))
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)
df = df.join(
    pd.DataFrame.sparse.from_spmatrix(
        mlb.fit_transform(df['tags']),
        index=df.index,
        columns=[x for x in mlb.classes_]))
# preprocessing
def clean_abstract(text):
    text = text.split()
    text = [x.strip() for x in text]
    text = [x.replace('\n', ' ').replace('\t', ' ') for x in text]
    text = ' '.join(text)
    text = re.sub('([.,!?()])', r' \1 ', text)
    return text

def get_texts(df):
    texts = 'multilabel classification: ' + df['texts_join'].apply(clean_abstract)
    texts = texts.values.tolist()
    return texts

def extract(label):
    l = sorted(ast.literal_eval(label))
    if not l:
        return 'none </s>'
    return ' , '.join(l) + ' </s>'

labels_li = [' '.join(x.lower().split()) for x in df.columns.to_list()[6:]]
label_dic = {}
number_dic = {}
for i, j in enumerate(labels_li):
    label_dic[j] = str(i)
    number_dic[str(i)] = j

label_dic['none'] = '102'
number_dic['102'] = 'none'


def get_labels(df):
    labels_li = [' '.join(x.lower().split()) for x in df.columns.to_list()[6:]]
    labels_matrix = np.array([labels_li] * len(df))

    mask = df.iloc[:, 6:].values.astype(bool)
    labels = []
    for l, m in zip(labels_matrix, mask):
        x = l[m]
        x = [label_dic[y] for y in x]
        if len(x) > 0:
            labels.append(', '.join(x) + ' </s>')
        else:
            labels.append('102 </s>')
    return labels


df = df.sort_values(by=['text_content_path'])

df_train, df_val, df_test = df.iloc[:60000], df.iloc[70000:75000], df.iloc[80000:]



class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = 't5-base'

        # data
        self.TOKENIZER = AutoTokenizer.from_pretrained("google/mt5-base")
        # T5Tokenizer.from_pretrained(self.MODEL_PATH)
        # AutoTokenizer.from_pretrained("google/mt5-base")
        #
        self.SRC_MAX_LENGTH = 1024
        self.TGT_MAX_LENGTH = 20
        self.BATCH_SIZE = 4
        self.VALIDATION_SPLIT = 0.25

        # model
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 1
        self.EPOCHS = 5

config = Config()

print("Done with config")

class T5Dataset(Dataset):
    def __init__(self, df, indices, set_type=None):
        super(T5Dataset, self).__init__()

        df = df.iloc[indices]
        self.texts = get_texts(df)
        self.set_type = set_type
        if self.set_type != 'test':
            self.labels = get_labels(df)

        self.tokenizer = config.TOKENIZER
        self.src_max_length = config.SRC_MAX_LENGTH
        self.tgt_max_length = config.TGT_MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        src_tokenized = self.tokenizer.encode_plus(
            self.texts[index],
            max_length=self.src_max_length,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        src_input_ids = src_tokenized['input_ids'].squeeze()
        src_attention_mask = src_tokenized['attention_mask'].squeeze()

        if self.set_type != 'test':
            tgt_tokenized = self.tokenizer.encode_plus(
                self.labels[index],
                max_length=self.tgt_max_length,
                pad_to_max_length=True,
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )
            tgt_input_ids = tgt_tokenized['input_ids'].squeeze()
            tgt_attention_mask = tgt_tokenized['attention_mask'].squeeze()

            return {
                'src_input_ids': src_input_ids.long(),
                'src_attention_mask': src_attention_mask.long(),
                'tgt_input_ids': tgt_input_ids.long(),
                'tgt_attention_mask': tgt_attention_mask.long()
            }

        return {
            'src_input_ids': src_input_ids.long(),
            'src_attention_mask': src_attention_mask.long()
        }

train_data = T5Dataset(df_train, list(range(len(df_train)))[:60000])
val_data = T5Dataset(df_val, list(range(len(df_val)))[:5000])

train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)

b = next(iter(train_dataloader))
for k, v in b.items():
    print(f'{k} shape: {v.shape}')



class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()

        self.t5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
        # MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
    #         T5ForConditionalGeneration.from_pretrained(config.MODEL_PATH)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            labels=None
    ):

        return self.t5_model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )


device = config.DEVICE



# the get_ohe function converts the decoder generated labels from textual -
# - format to a one hot encoded form, in order to calculate the micro f1 score

def get_ohe(x):
    #     labels_li = ['_'.join(x.lower().split()) for x in df_train.columns.to_list()[6:]]
    #     labels_li_indices = dict()
    #     for idx, label in enumerate(labels_li):
    #         labels_li_indices[label] = idx

    y = [labels.replace("<pad> ", "").replace("<pad>", "").replace(" ", "").split('</s>')[0].split(',') for labels in x]
    ohe = []
    for labels in y:
        temp = [0] * 103
        for label in labels:
            label = str(int(label))
            if label in number_dic:
                temp[int(label)] = 1
        ohe.append(temp)
    ohe = np.array(ohe)
    return ohe


def val(model, val_dataloader, criterion):

    val_loss = 0
    true, pred = [], []

    # set model.eval() every time during evaluation
    model.eval()

    for step, batch in enumerate(val_dataloader):
        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        b_tgt_input_ids = batch['tgt_input_ids']
        lm_labels = b_tgt_input_ids.to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # using torch.no_grad() during validation/inference is faster -
        # - since it does not update gradients.
        with torch.no_grad():
            # forward pass
            outputs = model(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask,
                labels=lm_labels,
                decoder_attention_mask=b_tgt_attention_mask)
            loss = outputs[0]

            val_loss += loss.item()

            # get true
            for true_id in b_tgt_input_ids:
                true_decoded = config.TOKENIZER.decode(true_id)
                true.append(true_decoded)

            # get pred (decoder generated textual label ids)
            pred_ids = model.t5_model.generate(
                input_ids=b_src_input_ids,
                attention_mask=b_src_attention_mask
            )
            pred_ids = pred_ids.cpu().numpy()
            for pred_id in pred_ids:
                pred_decoded = config.TOKENIZER.decode(pred_id)
                pred.append(pred_decoded)
    print("Pred first instance is " + str(pred[0]))
    print("True first instance is " + str(true[0]))
    true_ohe = get_ohe(true)
    pred_ohe = get_ohe(pred)
    print("True ohe is " + str(true_ohe[0]))
    print("Pred ohe is " + str(pred_ohe[0]))
    avg_val_loss = val_loss / len(val_dataloader)
    print('Val loss:', avg_val_loss)
    print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))

    val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
    print('Val micro f1 score:', val_micro_f1_score)
    return val_micro_f1_score


def train(
        model,
        train_dataloader,
        val_dataloader,
        criterion,
        optimizer,
        scheduler,
        epoch
):

    # we validate config.N_VALIDATE_DUR_TRAIN times during the training loop
    nv = config.N_VALIDATE_DUR_TRAIN
    temp = len(train_dataloader) // nv
    temp = temp - (temp % 100)
    validate_at_steps = [temp * x for x in range(1, nv + 1)]

    train_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader,
                                      desc='Epoch ' + str(epoch))):
        # set model.eval() every time during training
        model.train()

        # unpack the batch contents and push them to the device (cuda or cpu).
        b_src_input_ids = batch['src_input_ids'].to(device)
        b_src_attention_mask = batch['src_attention_mask'].to(device)

        lm_labels = batch['tgt_input_ids'].to(device)
        lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100

        b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)

        # clear accumulated gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids=b_src_input_ids,
                        attention_mask=b_src_attention_mask,
                        labels=lm_labels,
                        decoder_attention_mask=b_tgt_attention_mask)
        loss = outputs[0]
        train_loss += loss.item()

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # update scheduler
        scheduler.step()

        if step in validate_at_steps:
            print(f'-- Step: {step}')
            _ = val(model, val_dataloader, criterion)

    avg_train_loss = train_loss / len(train_dataloader)
    print('Training loss:', avg_train_loss)




def run():
    # setting a seed ensures reproducible results.
    # seed may affect the performance too.
    torch.manual_seed(config.SEED)

    criterion = nn.BCEWithLogitsLoss()

    # define the parameters to be optmized -
    # - and add regularization
    if config.FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_parameters, lr=config.LR)

    num_training_steps = len(train_dataloader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    max_val_micro_f1_score = float('-inf')
    for epoch in range(config.EPOCHS):
        train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epoch)
        val_micro_f1_score = val(model, val_dataloader, criterion)

        if config.SAVE_BEST_ONLY:
            if val_micro_f1_score > max_val_micro_f1_score:
                best_model = copy.deepcopy(model)
                best_val_micro_f1_score = val_micro_f1_score

                model_name = 'mt5_best_model_60000_t'
                torch.save(best_model.state_dict(), model_name + '.pt')

                print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')
                max_val_micro_f1_score = val_micro_f1_score

    return best_model, best_val_micro_f1_score




model = T5Model()
model.to(device);


best_model, best_val_micro_f1_score = run()