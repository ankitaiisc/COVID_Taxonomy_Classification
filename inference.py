import pickle, json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from stratified_kfold import create_train_test_splits, getstratifiedkfold
import pandas as pd 
import numpy as np
import os
from datasets import Dataset, Value, ClassLabel, Features
from transformers import DataCollatorWithPadding


def tokenize_function(entry):
    a = TOKENIZER(entry['text'])
    return {'input_ids':a['input_ids'], 'labels': entry['cat']}


from datasets import load_metric
from transformers import TrainingArguments, Trainer
from sklearn.metrics import classification_report

def compute_metrics(eval_preds):
    metric1 = load_metric("accuracy")
    metric2 = load_metric("f1")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return {'Accuracy': metric1.compute(predictions=predictions, references=labels)['accuracy'],
            'F1': metric2.compute(predictions=predictions, references=labels, average=None)['f1'].tolist(),
            'Weighted F1': metric2.compute(predictions=predictions, references=labels, average='weighted')['f1']}

def load_model_from_checkpoint(path, model):
    state_dict = torch.load(os.path.join(path, 'pytorch_model.bin'))
    model.load_state_dict(state_dict)
    return model

def get_best_checkpoint(input_dir):
    checkpoints = os.listdir(input_dir)
    checkpoints = [f for f in checkpoints if 'checkpoint' in f ]
    max_checkpoint_num = max([int(f.split('-')[-1]) for f in checkpoints])
    state_file = input_dir+'checkpoint-'+str(max_checkpoint_num)+'/trainer_state.json'
    with open(state_file, 'r') as f:
        state = json.loads(f.read())
    best_checkpoint = state['best_model_checkpoint']
    return best_checkpoint

#Evaluation
def recall_at_k(trues, preds_k, labels=None):
    category_wise_recall = {}
    category_support = {}
    labels = 'ABCDEFGHIK'
    
    for cat in set(trues):
        correct_count = 0
        total_count = 0
        for t, p_k in zip(trues, preds_k):
            if t!=cat:
                continue
            if t in p_k:
                correct_count+=1
            total_count+=1
        category_wise_recall[labels[cat]] = correct_count/total_count
        category_support[labels[cat]] = total_count

    count = 0
    for t, p_k in zip(trues, preds_k):
        if t in p_k:
            count+=1
    category_wise_recall['overall'] = count/len(trues)
    return category_wise_recall



base_model_dir = './models/CTBERT/full/'

TOKENIZER = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)
model = AutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', num_labels=10)

seed_dir = base_model_dir
best_checkpoint = get_best_checkpoint(seed_dir)
checkpoint_id = best_checkpoint.split('/')[-1]
best_checkpoint = seed_dir+checkpoint_id

model = load_model_from_checkpoint(best_checkpoint, model)
model.to('cuda:0')


topk=3
with torch.no_grad():
    preds, trues, probs = [], [], []
    for i, data in enumerate(tokenized_test, 0):
        inputs, labels = torch.tensor([data['input_ids']]).to('cuda:0'), torch.tensor([data['labels']]).to('cuda:0')
        outputs = model(inputs)
        correct = (outputs.logits.argmax(-1) == labels).sum().item()
        preds = preds + outputs.logits.argmax(-1).tolist()
        probs = probs + outputs.logits.tolist()

        preds_k = []
        for p in probs:
            preds_k.append(sorted(range(len(p)), key=lambda i: p[i])[-topk:])

        trues = trues + labels.tolist()
        
        
print(classification_report(trues, preds, target_names=[i for i in 'ABCDEFGHIK']))
category_wise_recall = recall_at_k(trues, preds_k)
print(category_wise_recall)