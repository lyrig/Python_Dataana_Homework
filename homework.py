import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emoji
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re, os



#从训练集中学习概率P(word)，和词的数目
def Learn(load_dir:str):
    ret = {}
    context = pd.read_csv(load_dir).to_dict('dict')
    text = context['lyrics']
    length = len(text)
    loop = tqdm(range(length), desc='Analysing Data')
        
    for index in loop:
        wordlist = text[index].split()
        for word in wordlist:
            ret.setdefault(word, 0)
            ret[word] += 1


    all = sum(list(ret.values()))
    for word in list(ret.keys()):
        ret[word] /= float(all)

    ret = dict(sorted(ret.items(), key=lambda x : x[1], reverse=True))
    print(type(ret))
    num = len(ret)
    return ret, num



'''
LogLinear的数据集
'''
class LoglinearSet(Dataset):
    def __init__(self, Load_dir:str, Worddict, dim:str='auto', typ:str = 'rock') -> None:
        super().__init__()
        self.data = pd.read_csv(Load_dir, encoding='utf-8').dropna(axis=1).to_dict('dict')
        self.len = len(self.data['lyrics'])
        self.wordfreq = Worddict
        self.wordnum = len(self.wordfreq)
        self.ty = typ
        if dim == 'auto':
            self.dim = int(self.wordnum * 0.8)
        elif dim.isnumeric():
            self.dim = int(dim)
        else:
            raise('Warning, dim Error')
            exit(1)
    def tokenizer(self, X:str):
        #print(self.wordfreq)
        #print(type(self.wordfreq))
        dictionary = list(dict(self.wordfreq).keys())
        feature = []
        #print(X)
        ret = []
        text = list(X.split(' '))
        for index in range(self.dim):
            if text.count(dictionary[index]):
                feature.append(text.count(dictionary[index]))
            else:
                feature.append(0)
        #print(len(feature))
        return torch.tensor(feature, dtype=torch.float)

    def __len__(self):
        return self.len
    

    def __getitem__(self, index):
        if self.data['playlist_genre'][index]==self.ty:
            ret =torch.tensor(1)
        else:
            ret = torch.tensor(0)
        return self.tokenizer(self.data['lyrics'][index]), ret



'''
Loglinear 模型
'''
class LogLinearModel(nn.Module):
    def __init__(self, Worddict:dict, dim:str='auto', out_channel:int=2):
        super().__init__()
        self.wordfreq = Worddict
        self.wordnum = len(self.wordfreq)
        if dim == 'auto':
            self.dim = int(self.wordnum * 0.8)
        elif dim.isnumeric():
            self.dim = int(dim)
        else:
            raise('Warning, dim Error')
            exit(1)
        
        self.fc1 = nn.Linear(in_features=self.dim, out_features=out_channel)

    def tokenizer(self, X:str):
        #print(self.wordfreq)
        #print(type(self.wordfreq))
        dictionary = list(dict(self.wordfreq).keys())
        feature = []
        #print(X)
        ret = []
        for i in X:
            text = list(X.split(' '))
            for index in range(self.dim):
                feature.append(text.count(dictionary[index]))
        
        return torch.tensor(feature)

    def forward(self, X):
        out = self.fc1(X)
        out = F.softmax(out)

        return out

if __name__ == '__main__':
    worddict, wordnum = Learn('./data/songs.csv')
    set = LoglinearSet('./data/songs.csv', worddict)
    setpop = LoglinearSet('./data/songs.csv', worddict, typ='pop')
    setrap = LoglinearSet('./data/songs.csv', worddict, typ='rap')\
    
    loader = DataLoader(set, batch_size=128, shuffle=True)
    loader2 = DataLoader(setpop, batch_size=128, shuffle=True)
    loader3 = DataLoader(setrap, batch_size=128, shuffle=True)


    modelrock = LogLinearModel(worddict)
    modelpop = LogLinearModel(worddict)
    modelrap = LogLinearModel(worddict)
    #print(set[0])
    loop = tqdm(enumerate(loader),total=len(loader) )
    optrock = optim.Adam(modelrock.parameters(), lr = 1e-3)
    optpop = optim.Adam(modelpop.parameters(), lr = 1e-3)
    optrap = optim.Adam(modelrap.parameters(), lr = 1e-3)


    criterion = nn.CrossEntropyLoss()
    last = 0
    History = {'error':[], 'accuracy':[], 'max_error':[], 'error_id':[]}
    for epoch in range(10):
        for i, (input, sexist_label) in loop:
            print(input.shape)
            print(input)
            outrock = modelrock(input)

            loss1 = criterion(outrock, sexist_label)
            loss1.backward()
            optrock.step()
            optrock.zero_grad()

            if i % 5 == 0:
                out = torch.argmax(outrock)
                acc = (out == sexist_label).sum().item()/len(sexist_label)
                print('Acc:'+ str(acc))

    loop = tqdm(enumerate(loader2),total=len(loader2) )
    for epoch in range(10):
        for i, (input, sexist_label) in loop:
            
            outpop = modelpop(input)
            
            
            #print(outrock.shape)

            loss2 = criterion(outpop, sexist_label)
            loss2.backward()
            optpop.step()
            optpop.zero_grad()

            if i % 5 == 0:
                out = torch.argmax(outpop)
                acc = (out == sexist_label).sum().item()/len(sexist_label)
                print('Acc:'+ str(acc))

    loop = tqdm(enumerate(loader3),total=len(loader3) )
    for epoch in range(10):
        for i, (input, sexist_label) in loop:
            outrap = modelrap(input)
            
            
            #print(outrock.shape)

            loss3 = criterion(outrock, sexist_label)
            loss3.backward()
            optrap.step()
            optrap.zero_grad()

            if i % 5 == 0:
                out = torch.argmax(outrap)
                acc = (out == sexist_label).sum().item()/len(sexist_label)
                print('Acc:'+ str(acc))


    torch.save(modelrock, './model/rock.pt')
    torch.save(modelpop, './model/pop.pt')
    torch.save(modelrap, './model/rap.pt')
