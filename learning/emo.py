
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, TrainingArguments, Trainer, \
    AdapterType, AdapterTrainer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaForSequenceClassification
from transformers import RobertaAdapterModel, DistilBertForSequenceClassification
from transformers import RobertaConfig, RobertaModelWithHeads

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.special import softmax
import numpy as np
import torch
from torch import nn

# The emotion dataset in the Hugging Face datasets library has the following format:
''' 6: 'sadness' 'joy' 'love'  'anger' 'fear' 'surprise 
orignal model 7:  anger  disgust  fear  joy  neutral  sadness  surprise 
new data set has love, while old data set has disgust, nuetral
'''
num_labels = 6  # Set this to the number of classes in your new data


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)

    # Ensure we have probabilities for each class
    num_classes = len(set(labels))
    if probs.shape[1] != num_classes:
        probs_resized = np.zeros((probs.shape[0], num_classes))
        probs_resized[:, :probs.shape[1]] = probs
        probs = probs_resized

    return {'roc_auc': roc_auc_score(labels, probs, multi_class='ovo'),
            'precision': precision_score(labels, predictions, average='weighted'),
            'recall': recall_score(labels, predictions, average='weighted'),
            'f1': f1_score(labels, predictions, average='weighted'),
            }

class AdapterModel(nn.Module):
    def __init__(self):
        super(AdapterModel, self).__init__()

        # Load the pre-trained model as a feature extractor (with frozen weights)
        self.base_model = RobertaAdapterModel.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Insert an adapter layer to map from 7 to 6 classes
        self.adapter = nn.Linear(7, 6)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # Use the adapter to transform the outputs to the required number of classes
        adapter_outputs = self.adapter(outputs.logits)

        return adapter_outputs


# Load the emotion dataset
dataset = load_dataset('emotion')

config = RobertaConfig.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base",
)

# Load the base model
model = RobertaModelWithHeads.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base",
    config=config,
)

model = DistilBertForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base', num_labels=7)


# Add an adapter
model.add_adapter("emotion")
# Define our custom head
class CustomHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 7)
        self.activation = nn.ReLU()
        self.out_proj = nn.Linear(7, 6)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return x

# model.add_custom_head("emotion", CustomHead())
# model.set_active_classification_head(CustomHead(), "emotion")
# Train the adapter
model.train_adapter("emotion")
model.set_active_adapters("emotion")
adapter_output_size = 7 #model.config.adapters.adapter_config["emotion"]["hidden_size"]
print (adapter_output_size)
model.config.adapters.get(adapter_name="emotion").head = nn.Linear(adapter_output_size, 6)

#model.config.adapters.adapter_config["emotion"]["output_size"] = 6

#model.base_model.set_output_embeddings(nn.Linear(adapter_output_size, 6))

# model.base_model.get_adapter("emotion").fc_out = nn.Linear(adapter_output_size, 6)

#model.config.adapters.adapter_dict["emotion"].fc_out = nn.Linear(adapter_output_size, 6)

# model.adapter_dict["emotion"].adapter_head = nn.Linear(7, 6)
#
# Freeze all parameters of the model except for the adapter
model.freeze_model()

# Load the pre-trained DistilRoberta model
tokenizer = RobertaTokenizerFast.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
# https://towardsdatascience.com/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1

# config = RobertaConfig.from_pretrained('j-hartmann/emotion-english-distilroberta-base')
# config.num_labels = num_labels
# model = RobertaForSequenceClassification.from_pretrained('j-hartmann/emotion-english-distilroberta-base', config=config)


# Preprocessing the data
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

train_dataset, val_dataset, test_dataset = dataset['train'], dataset['validation'], dataset['test']
train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))
val_dataset = val_dataset.map(tokenize, batched=True, batch_size=len(val_dataset))
test_dataset = test_dataset.map(tokenize, batched=True, batch_size=len(test_dataset))
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Adjust these values as needed
num_train_epochs = 5
learning_rate = 1e-5
warmup_steps = int(len(train_dataset) * num_train_epochs * 0.1)  # 10% of train data for warm-up
total_steps = len(train_dataset) * num_train_epochs

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_dir='./logs',
    use_mps_device=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# Create a learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Define the Trainer
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, lr_scheduler),
)



# Train the model
trainer.train()

eval_results = trainer.evaluate(test_dataset)
print(eval_results)



