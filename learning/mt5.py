import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from transformers import (
    T5Model,
    get_linear_schedule_with_warmup,
    MT5ForConditionalGeneration, AutoTokenizer
)

def get_texts(df):
    texts = df['text']
    texts = texts.values.tolist()
    return texts

def convertLanguage(s):
    n = sorted([tag_id_map[x] for x in s])
    return ', '.join(n)


def get_labels(df):
    labels = df.apply(lambda x: convertLanguage(x.applied_tags), axis = 1)
    labels = labels.values.tolist()
    return labels

df = pd.read_csv("web_view_ad_landing_page_annotation_gcs_prod.csv", sep='\t')
df.applied_tags = df.applied_tags.apply(lambda x: x.split(','))
tags = set()
for i in df.applied_tags:
    for x in i:
        tags.add(x)

tag_id_map = {}
id_tag_map = {}
for i, j in enumerate(sorted(list(tags))):
    tag_id_map[j] = str(i)
    id_tag_map[str(i)] = j
# save tag_id_map to disk
import json
with open('tag_id_map.json', 'w') as fp:
    json.dump(tag_id_map, fp)


print(id_tag_map)

df_train = df[df.created_at < "2023-02-28"].head(600)
# save df
df_train.to_csv("train.csv", sep='\t', index=False)

df_val = df[(df.created_at > "2023-02-28") & (df.created_at < "2023-03-15")].head(100)
# df_test = df[(df.created_at > "2023-03-15")]
# df_train = df[:60]
# df_val = df[60:80]
# df_test = df[80:]

class Config:
    def __init__(self):
        super(Config, self).__init__()

        self.SEED = 42
        self.MODEL_PATH = 'mt5-base'

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
        # self.DEVICE = torch.device("mps") for mac has memory limit.
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.FULL_FINETUNING = True
        self.LR = 3e-5
        self.OPTIMIZER = 'AdamW'
        self.CRITERION = 'BCEWithLogitsLoss'
        self.SAVE_BEST_ONLY = True
        self.N_VALIDATE_DUR_TRAIN = 1

        self.EPOCHS = 5 # more epochs for better perf
config = Config()

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

train_data = T5Dataset(df_train, list(range(len(df_train))))
val_data = T5Dataset(df_val, list(range(len(df_val))))
train_dataloader = DataLoader(train_data, batch_size=config.BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=config.BATCH_SIZE)


class T5Model(nn.Module):
    def __init__(self):
        super(T5Model, self).__init__()
        self.t5_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-base")
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
print(device)

def get_ohe(x):
    ohe = []
    for l in x:
        # print(config.TOKENIZER.encode(l))
        t = [x for x in config.TOKENIZER.encode(l) if (x != 1 and x != 0 and x != 261)]
        temp = [0] * 102
        for z in t:
            label = config.TOKENIZER.decode(z)
            if label in id_tag_map:
                temp[int(label)] = 1
        ohe.append(temp)
    ohe = np.array(ohe)
    return ohe
# def val(model, val_dataloader, criterion):
#
#     val_loss = 0
#     true, pred = [], []
#
#     # set model.eval() every time during evaluation
#     model.eval()
#
#     for step, batch in enumerate(val_dataloader):
#         # unpack the batch contents and push them to the device (cuda or cpu).
#         b_src_input_ids = batch['src_input_ids'].to(device)
#         b_src_attention_mask = batch['src_attention_mask'].to(device)
#
#         b_tgt_input_ids = batch['tgt_input_ids']
#         lm_labels = b_tgt_input_ids.to(device)
#         lm_labels[lm_labels[:, :] == config.TOKENIZER.pad_token_id] = -100
#
#         b_tgt_attention_mask = batch['tgt_attention_mask'].to(device)
#
#         # using torch.no_grad() during validation/inference is faster -
#         # - since it does not update gradients.
#         with torch.no_grad():
#             # forward pass
#             outputs = model(
#                 input_ids=b_src_input_ids,
#                 attention_mask=b_src_attention_mask,
#                 labels=lm_labels,
#                 decoder_attention_mask=b_tgt_attention_mask)
#             loss = outputs[0]
#
#             val_loss += loss.item()
#             print(b_tgt_input_ids)
#             # get true
#             for true_id in b_tgt_input_ids.squeeze():
#                 print(true_id)
#                 # if true_id >= 0:
#                 true_decoded = config.TOKENIZER.decode(true_id)
#                 true.append(true_decoded)
#
#             # get pred (decoder generated textual label ids)
#             pred_ids = model.t5_model.generate(
#                 input_ids=b_src_input_ids,
#                 attention_mask=b_src_attention_mask
#             )
#             pred_ids = pred_ids.cpu().numpy()
#             for pred_id in pred_ids:
#                 pred_decoded = config.TOKENIZER.decode(pred_id)
#                 pred.append(pred_decoded)
#     print("Pred first instance is " + str(pred[0]))
#     print("True first instance is " + str(true[0]))
#     true_ohe = get_ohe(true)
#     pred_ohe = get_ohe(pred)
#     print("True ohe is " + str(true_ohe[0]))
#     print("Pred ohe is " + str(pred_ohe[0]))
#     avg_val_loss = val_loss / len(val_dataloader)
#     print('Val loss:', avg_val_loss)
#     print('Val accuracy:', accuracy_score(true_ohe, pred_ohe))
#
#     val_micro_f1_score = f1_score(true_ohe, pred_ohe, average='micro')
#     print('Val micro f1 score:', val_micro_f1_score)
#     return val_micro_f1_score

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
            for true_id in b_tgt_input_ids.squeeze():
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

                model_name = 'mt5_best_model_test'
                torch.save(best_model.state_dict(), model_name + '.pt')

                print(f'--- Best Model. Val loss: {max_val_micro_f1_score} -> {val_micro_f1_score}')
                max_val_micro_f1_score = val_micro_f1_score

    return best_model, best_val_micro_f1_score

model = T5Model()
model.to(device)

best_model, best_val_micro_f1_score = run()

# test_data = T5Dataset(df_test, list(range(len(df_test))))
# test_dataloader = DataLoader(test_data, batch_size=config.BATCH_SIZE)