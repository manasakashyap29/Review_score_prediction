import os
import math
import sys
import io
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import numpy as np

import time
import warnings

warnings.filterwarnings("ignore")


start = time.time()

######-----FILES------######

test_file = "drugsComTest_raw.tsv"
model_file = "drug_review_score_pred_model"

############################
random_seed = 123

torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True

stopwords_list = ['a', 'about', 'across', 'all', 'already', 'also', 'an', 'and', 'any', 'anybody', 'anyone', 'anything', 'anywhere', 'are', 'area', 'areas', 'around', 'as', 'ask', 'asked', 'asking', 'asks', 'at', 'be', 'been', 'before', 'being', 'beings', 'by', 'came', 'come', 'did', 'do', 'does', 'during', 'each', 'either', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'face', 'faces', 'for', 'four', 'from', 'further', 'furthered', 'furthering', 'furthers', 'gave', 'general', 'generally', 'get', 'gets', 'give', 'given', 'gives', 'go', 'got', 'group', 'grouped', 'grouping', 'groups', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'herself', 'him', 'himself', 'his', 'how', 'however', 'if', 'in', 'is', 'it', 'its', 'itself', 'just', 'keep', 'keeps', 'let', 'lets', 'made', 'make', 'making', 'man', 'many', 'me', 'member', 'members', 'men', 'mr', 'mrs', 'much', 'my', 'myself', 'number', 'numbers', 'of', 'one', 'or', 'other', 'others', 'our', 'per', 'place', 'places', 'put', 'puts', 'room', 'rooms', 'saw', 'say', 'says', 'second', 'seconds', 'see', 'seem', 'seemed', 'seeming', 'seems', 'sees', 'several', 'shall', 'she', 'should', 'side', 'sides', 'some', 'somebody', 'someone', 'something', 'somewhere', 'state', 'states', 'still', 'such', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'thing', 'things', 'this', 'those', 'though', 'thought', 'thoughts', 'three', 'through', 'thus', 'to', 'today', 'too', 'took', 'two', 'us', 'was', 'way', 'ways', 'we', 'wells', 'went', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whose', 'why', 'with', 'would', 'year', 'years', 'you', 'your', 'yours']
punct_list = ["#", "&", ".", ",", "(", ")", ":", ";", "/", "?", "'"]

def isValid(word):
    if "_" in word:
        return False
    if re.fullmatch("[A-Za-z][a-z]+[A-Z]+[a-z]*", word):
        return False
    if any(x in word for x in punct_list):
        return True
    if word.lower() in stopwords_list:
        return False
    if len(word) == 1:
        return False  
    return True


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout,embed_index):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, n_filters)
        self.fc2 = nn.Linear(n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device('cpu')
        self.embedding_index_dict = embed_index
        
    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))    
        res_fc1 = self.fc1(cat)

        return self.fc2(res_fc1)
    

def load_data_files(data_file):
    documents = []
    labels = []

    data = pd.read_csv(data_file, sep="\t", header=0)

    for index, row in data.iterrows():
        doc = [word for word in row['review'].strip().strip('"').split() if isValid(word)]
        documents.append(doc)

        label = float(row['rating'])/10
        labels.append(label)
    
    return documents, labels

def accuracy(preds, y):
    #Rounded to nearest 0.1
    rounded_preds = torch.round(torch.sigmoid(preds) * 10)/10

    correct = (abs(rounded_preds - y)<0.2).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc.item()


def test_model(test_file, model_file):
    documents, true_labels = load_data_files(test_file)
    
    model = torch.load(model_file)
    model.eval()

    padding_size = max(model.filter_sizes)
    predictions = []

    with torch.no_grad():
        for doc in documents:
            indexed_doc = []
            for word in doc:
                word_index = -1
                try:
                    word_index = model.embedding_index_dict[word]
                except:
                    word_index = model.embedding_index_dict['<pad>']
                indexed_doc.append(word_index)
            
            padding = [model.embedding_index_dict['<pad>'] for i in range(padding_size)]
            indexed_doc = torch.LongTensor(padding + indexed_doc + padding).to(model.device).unsqueeze(1)
            prediction = model(indexed_doc)
            predictions.append(prediction)

    print('Finished Testing. Time:', time.time() - start)
    print("Test Accuracy:", accuracy(torch.FloatTensor(predictions), torch.FloatTensor(true_labels)))

if __name__ == "__main__":
    #Test
    test_model(test_file, model_file)