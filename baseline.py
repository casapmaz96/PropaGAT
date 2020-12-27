import torch.nn as nn
import torch.nn.functional as F
from data_loader import DataLoader
import torch
from random import shuffle
import time
from sklearn.metrics import f1_score
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer, BertModel

###The RNN code below is from the official PyTorch tutorial on RNNs by Sean Robertson (2017).###
###Tutorial link: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html ###

class Baseline(nn.Module):
    def __init__(self, edim, vocab,  hidden_size, output_size, useBert=0):
        super(Baseline, self).__init__()
        self.useBert = useBert
        self.hidden_size = hidden_size
        if useBert==0:
            self.embed = nn.Embedding(vocab, edim)
            self.edim=edim
        else: self.edim = 768
 #       print('1')
        self.rnn = nn.RNN(input_size=self.edim, hidden_size=hidden_size, batch_first=True, num_layers=2)
 #       print('2')
        self.nlinearity = nn.LeakyReLU(negative_slope=0.01)
        self.linear = nn.Linear(hidden_size, output_size)
#        print('6')
        #self.softmax = F.oftmax(dim=1)

    def forward(self, input):
        if self.useBert == 0:
            words = self.embed(input)
        else: words = input
      #  print('3')
        seq, _ = self.rnn(words)
       # print('4')
        outp = self.nlinearity(self.linear(seq[:,-1,:]))
        outp = F.softmax(outp, dim=1)
        return outp



def trainBaseline(trainPath, vocabPath, labelPath, epoch=1, useBert=0, load=0, modelPath = '', saveName = "baseline"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dl = DataLoader(vocabPath, labelPath)

    if useBert == 0:
        trainArticles = dl.readData_ub(trainPath)
    else:
        trainArticles = dl.readData_bert(trainPath)
    sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, dl.noSample, replacement=False))
    lenArt = dl.lenArt
        #print('epoch: ', i)


    baseline = Baseline(64, dl.voc.keysize, 32, 2, useBert = useBert)
#    print('7')
    docLoss = nn.CrossEntropyLoss()
#    print('5')
    opt = torch.optim.Adam(baseline.parameters(), lr = 0.001)

    if load == 1:
        checkpoint = torch.load(modelPath)

        baseline.load_state_dict(checkpoint['baseline_model'])
        #DGAT.load_state_dict(checkpoint['document_model'])
        opt.load_state_dict(checkpoint['optimizer'])

    if useBert==1:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    ##Iterate over articles in dataset
    for e in range(0, epoch):
        truelabels = []; predictions = []

        for n, ind in enumerate(sample):

            a = trainArticles[ind]
            id, article, l = a
            aTemp = torch.tensor([])
            if l>0:
                l=torch.tensor([1], dtype=torch.long) #.to('cuda')
                truelabels.append(1)
            else:
                l=torch.tensor([0], dtype=torch.long) #.to('cuda')
                truelabels.append(0)

            for i, s in enumerate(article):
                if useBert == 1:
                    sTemp = bert(torch.tensor(tokenizer(s[:510] if len(s)>510 else s)['input_ids'], device=device).unsqueeze(0))[0].detach().squeeze(dim=0).to('cpu')
                else: sTemp = s
                aTemp =torch.cat((aTemp, sTemp.float()), dim=0)


           #if i == 0:
            #h = torch.zeros((1, 32))
            label = baseline(aTemp.unsqueeze(0))

            loss = docLoss(label, l)

            if l.item() == label.argmax().item():
                if l.item() == 0:
         #           tneg += 1
                    predictions.append(0)
                else:
          #          tpos += 1
                    predictions.append(1)
            else:
                if l.item() == 0:
          #          fpos += 1
                    predictions.append(1)
                else:
          #          fneg += 1
                    predictions.append(0)
            opt.zero_grad()
            loss.backward()
            opt.step()

        print('f1 score: ', f1_score(truelabels, predictions, average= None))

   # if debugMode == 0:
    torch.save({'baseline_model': baseline.state_dict(),
                'optimizer': opt.state_dict()

               }, 'dataset/{}.tar'.format(saveName))

#        print(a)
#        break

def testBaseline(testPath, vocabPath, labelPath, useBert=0, modelPath = ''):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dl = DataLoader(vocabPath, labelPath)

    if useBert == 0:
        testArticles = dl.readData_ub(testPath)
    else:
        testArticles = dl.readData_bert(testPath)
    #sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, dl.noSample, replacement=False))
    lenArt = dl.lenArt
        #print('epoch: ', i)


    baseline = Baseline(64, dl.voc.keysize, 32, 2, useBert = useBert)
#    print('7')
    docLoss = nn.CrossEntropyLoss()
#    print('5')
#    opt = torch.optim.Adam(baseline.parameters(), lr = 0.001)

#    if load == 1:
    checkpoint = torch.load(modelPath)

    baseline.load_state_dict(checkpoint['baseline_model'])
        #DGAT.load_state_dict(checkpoint['document_model'])
      #  opt.load_state_dict(checkpoint['optimizer'])

    if useBert==1:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    ##Iterate over articles in dataset
    #for e in range(0, epoch):
    truelabels = []; predictions = []

    for n, a in enumerate(testArticles):
        id, article, l = a
        aTemp = torch.tensor([])
        if l>0:
            l=torch.tensor([1], dtype=torch.long) #.to('cuda')
            truelabels.append(1)
        else:
            l=torch.tensor([0], dtype=torch.long) #.to('cuda')
            truelabels.append(0)

        for i, s in enumerate(article):
            if useBert == 1:
                sTemp = bert(torch.tensor(tokenizer(s[:510] if len(s)>510 else s)['input_ids'], device=device).unsqueeze(0))[0].detach().squeeze(dim=0).to('cpu')
            else: sTemp = s
            aTemp =torch.cat((aTemp, sTemp.float()), dim=0)


           #if i == 0:
            #h = torch.zeros((1, 32))
        label = baseline(aTemp.unsqueeze(0))

#        loss = docLoss(label, l)

        if l.item() == label.argmax().item():
            if l.item() == 0:
         #           tneg += 1
                predictions.append(0)
            else:
          #          tpos += 1
                predictions.append(1)
        else:
            if l.item() == 0:
          #          fpos += 1
                predictions.append(1)
            else:
          #          fneg += 1
                predictions.append(0)


    print('f1 score: ', f1_score(truelabels, predictions, average= None))


