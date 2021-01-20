import numpy as np
import csv
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import torch
import time
import glob 
#from streamlit import caching
import itertools
#import String.Split
def preprocessor(datasetTrain, datasetTest, vocDict={'<UNK>':0}, path='dataset'):
    ##Create txt files where each file is an article and each  line is a sentence
    ##Create vocab dictionary (and save)

    nltk.download('punkt')

    articlesTrain = {}
    with open(datasetTrain, 'r') as data:

        print("Processing the training set\n")
        reader = csv.reader(data, delimiter='\t')
        i = 0

        for a in reader:
            articlesTrain[a[4]] = sent_tokenize(a[0])

        with open('articlesTrain.json', 'w') as artfile:
            json.dump(articlesTrain, artfile)

    print("Processing the test set\n")
    articlesTest={}
    with open(datasetTest) as data:
        reader = csv.reader(data, delimiter='\t')
        for a in reader:
            articlesTest[a[4]] = sent_tokenize(a[0])

    with open('articlesTest.json', 'w') as artfile:
        json.dump(articlesTest, artfile)

        
def sentencePreprocessor(trainPath, testPath, trainLabels, testLabels, saveTrain = 'trainSentences.json', saveTest = 'testSentences.json'):

    ##The same preprocessing method as above, but for dataset annotated at sentence level
    
    ##Create labels dictionary
    labelDict = {}

    with open(trainLabels) as l:
        lr = l.readlines()
        for line in lr:
            #print(line.split('\t'))
            lineSplit = line.split('\t')
            if lineSplit[0] in labelDict.keys():
                p = 0
                if lineSplit[2][0] == 'p':
                    p = 1
                else: p = 0
                labelDict[lineSplit[0]].append(p)
            else:
                p = 0
                if lineSplit[2][0] == 'p':
                    p = 1
                else: p = 0

                labelDict[lineSplit[0]] = [p]

    with open(testLabels) as l:
        lr = l.readlines()
        for line in lr:
            #print(line.split('\t'))
            lineSplit = line.split('\t')
            if lineSplit[0] in labelDict.keys():
                p = 0
                if lineSplit[2][0] == 'p':
                    p = 1
                else: p = 0

                labelDict[lineSplit[0]].append(p)
            else:
                p = 0
                if lineSplit[2][0] == 'p':
                    p = 1
                else: p = 0

                labelDict[lineSplit[0]] = [p]


    with open('sentenceLabels.json', 'w') as senFile:
        json.dump(labelDict, senFile)

    ##Create train articles json file

    trfiles = glob.glob(trainPath+'/*.txt')
    trainSentences = {}
    for f in trfiles:
        fl = open(f)
        lines = fl.readlines()
#        print(len(lines[1]))
#        break
        trainSentences[f[len(trainPath)+8:-4]] = []
        for l in lines:
            if len(l) > 1:
                trainSentences[f[len(trainPath)+8:-4]].append(l.rstrip('\n'))
            else: trainSentences[f[len(trainPath)+8:-4]].append(l)
    with open('sentenceArticlesTrain.json', 'w') as senFile:
        json.dump(trainSentences, senFile)

    ##Create train articles json file

    tsfiles = glob.glob(testPath+'/*.txt')
    testSentences = {}
    for f in tsfiles:
        fl = open(f)
        lines = fl.readlines()
#        print(len(lines[1]))
#        break
        testSentences[f[len(testPath)+8:-4]] = []
        for l in lines:
            if len(l) > 1:
                testSentences[f[len(testPath)+8:-4]].append(l.rstrip('\n'))
            else: testSentences[f[len(testPath)+8:-4]].append(l)
    with open('sentenceArticlesTest.json', 'w') as senFile:
        json.dump(testSentences, senFile)


class Vocab:

    def __init__(self, dict={'<UNK>':0}):

        self.w2i = dict
        self.keysize = len(list(self.w2i.keys()))

    def addWords(self, text):
        #text = string sentence

        words = word_tokenize(text)
        words = [w.lower() for w in words if w.isalpha()]

        for w in words:
            if w not in list(self.w2i.keys()):
                self.w2i[w] = self.keysize
                self.keysize += 1

    def s2v(self, text, device=torch.device('cpu')):
        words = word_tokenize(text)
        #words = [w.lower() for w in words if w.isalpha()]
        sentence = torch.tensor([], device=device)

#        if '<EOS>' in list(self.w2i.keys()):
#            eos = self.w2i['<EOS>']
#        else:
#            self.w2i['<EOS>'] = self.keysize
#            eos = self.w2i['<EOS>']

        for i, w in enumerate(words):
            if w.isalpha():
                w = w.lower()
                if w in list(self.w2i.keys()):
                    sentence = torch.cat((sentence, torch.tensor([self.w2i[w]], device=device).float()), 0)
                else:
                    sentence = torch.cat((sentence, torch.tensor([0], device=device).float()), 0)
#        sentence = torch.cat((sentence, torch.tensor([eos], device=torch.device('cuda')).float()), 0 )
        return sentence



