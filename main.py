#!/usr/bin/env python
# coding: utf-8


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


df = pd.read_csv('covid_facts.csv')
df['cat'] = df['Category'].astype(str).str[0]
df['text'] = df['Claim']
df = df[['text','cat']]

#kfolddata = create_train_test_splits(5, df)

base_model_dir = './models/CTBERT/fold_5e6_'
topk = 3
for fold_num in np.arange(5):
    
    TOKENIZER = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert-v2")
    model = AutoModelForSequenceClassification.from_pretrained('digitalepidemiologylab/covid-twitter-bert-v2', num_labels=10)
    
    seed_dir = base_model_dir+str(fold_num)+'/'
    best_checkpoint = get_best_checkpoint(seed_dir)
    checkpoint_id = best_checkpoint.split('/')[-1]
    best_checkpoint = seed_dir+checkpoint_id
    
    model = load_model_from_checkpoint(best_checkpoint, model)
    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER)

    #one time: create data splits
    #df_train, df_test = kfolddata[fold_num]['train'], kfolddata[fold_num]['test']
    
    #os.makedirs('./data/fold_'+str(fold_num), exist_ok=True)
    #df_train.to_csv('./data/fold_'+str(fold_num)+'/train_'+str(fold_num)+'.csv')
    #df_test.to_csv('./data/fold_'+str(fold_num)+'/test_'+str(fold_num)+'.csv')
    
    df_train = pd.read_csv('./data/fold_'+str(fold_num)+'/train_'+str(fold_num)+'.csv')
    df_test = pd.read_csv('./data/fold_'+str(fold_num)+'/test_'+str(fold_num)+'.csv')

    train_dataset = Dataset.from_pandas(df_train)
    train_dataset = train_dataset.class_encode_column("cat")

    test_dataset = Dataset.from_pandas(df_test)
    test_dataset = test_dataset.class_encode_column("cat")
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir='./models/CTBERT/'+'fold_5e6_'+str(fold_num),
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=5,
        num_train_epochs=20,
        warmup_steps=2, 
        load_best_model_at_end=True,
        metric_for_best_model='Weighted F1',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-6,
        lr_scheduler_type='linear',
    )


    trainer = Trainer(
        model=model.to('cuda:0'),
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=TOKENIZER,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    #uncomment following line to train a your own model.
    #trainer.train()
    
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
            
    print(classification_report(labels, preds, target_names=[i for i in 'ABCDEFGHIK']))
    
#     with open('./results/CTBERT/'+'true_5e6_'+str(fold_num)+'.pickle', 'wb') as handle:
#         pickle.dump(trues, handle)
        
#     with open('./results/CTBERT/'+'pred_5e6_'+str(fold_num)+'.pickle', 'wb') as handle:
#         pickle.dump(preds, handle)

#       with open('./results/CTBERT/fine-grained/'+'pred_k_'+str(fold_num)+'.pickle', 'wb') as handle:
#         pickle.dump(preds_k, handle)
        
    
    del model


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




all_trues = []
all_preds_k = []
all_preds = []
for fold_num in range(5):
    with open('./results/CTBERT/'+'true_5e6_'+str(fold_num)+'.pickle', 'rb') as handle:
        trues = pickle.load(handle)

    with open('./results/CTBERT/'+'pred_k_5e6_'+str(fold_num)+'.pickle', 'rb') as handle:
        preds_k = pickle.load(handle)
        
    with open('./results/CTBERT/'+'pred_5e6_'+str(fold_num)+'.pickle', 'rb') as handle:
        preds = pickle.load(handle)
        
    all_trues+=trues
    all_preds_k+=preds_k
    all_preds+=preds

category_wise_recall = recall_at_k(all_trues, all_preds_k)
print(category_wise_recall)
