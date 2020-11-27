import csv
import torch
import torch.utils.data as data
from preprocessing import Vocab
import numpy as np
import glob, os
from random import shuffle
import json
class DataLoader:

    def __init__(self, vocPath, labelsPath):

        #vocab = np.load(vocPath, allow_pickle=True)[()]
        with open(vocPath, 'r') as vf:
            vocab = json.load(vf)
            self.voc = Vocab(vocab)

        with open(labelsPath, 'r') as lf:
            self.labels = json.load(lf) #np.load(labelsPath, allow_pickle=True)[()]

    def readData(self, trainPath):


#        """        ##READ LINES
#        trainFiles = glob.glob(trainPath+'/*.txt')
#        testFiles = glob.glob(testPath+'/*.txt')
#        shuffle(testFiles); shuffle(trainFiles)

#        trarticles = []; tsarticles = []
#        i = 0

#        for f in trainFiles:
#            tf = open(f)
#            lines = tf.readlines()
#            lines = map(lambda x: self.voc.s2v(x).long().to('cuda'), lines) #[self.voc.s2v(l).long() for l in lines]
#            #print(f)
#            trarticles.append( (lines, int(self.labels[f[len(trainPath)+1:-4]])) )
#            #i+=1
#            tf.close()
#        i=0
#        for f in testFiles:
#            tf = open(f)
#            lines = tf.readlines()
#            lines = map(lambda x: self.voc.s2v(x).long().to('cuda'), lines) # [self.voc.s2v(l).long() for l in lines]
#            tsarticles.append( (lines, int(self.labels[f[len(testPath)+1:-4]])) )
#            #i+=1
#            tf.close()"""
        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)
        #with open(testPath, 'r') as tsfile:
        #    testArticles = json.load(tsfile)

        trarticles = [(map(lambda x: self.voc.s2v(x).long().to(torch.device('cuda')), trainArticles[a]), int(self.labels[a])) for a in trainArticles.keys()]
        #tsarticles = [(map(lambda x: self.voc.s2v(x).long().to(torch.device('cuda')), testArticles[a]), int(self.labels[a])) for a in testArticles.keys()]
        return trarticles

class DataLoaderv2:

    def __init__(self, vocPath, labelsPath):

        #vocab = np.load(vocPath, allow_pickle=True)[()]
        with open(vocPath, 'r') as vf:
            vocab = json.load(vf)
            self.voc = Vocab(vocab)

        with open(labelsPath, 'r') as lf:
            self.labels = json.load(lf) #np.load(labelsPath, allow_pickle=True)[()]

    def readData(self, trainPath):


#        """        ##READ LINES
#        trainFiles = glob.glob(trainPath+'/*.txt')
#        testFiles = glob.glob(testPath+'/*.txt')
#        shuffle(testFiles); shuffle(trainFiles)

#        trarticles = []; tsarticles = []
#        i = 0

#        for f in trainFiles:
#            tf = open(f)
#            lines = tf.readlines()
#            lines = map(lambda x: self.voc.s2v(x).long().to('cuda'), lines) #[self.voc.s2v(l).long() for l in lines]
#            #print(f)
#            trarticles.append( (lines, int(self.labels[f[len(trainPath)+1:-4]])) )
#            #i+=1
#            tf.close()
#        i=0
#        for f in testFiles:
#            tf = open(f)
#            lines = tf.readlines()
#            lines = map(lambda x: self.voc.s2v(x).long().to('cuda'), lines) # [self.voc.s2v(l).long() for l in lines]
#            tsarticles.append( (lines, int(self.labels[f[len(testPath)+1:-4]])) )
#            #i+=1
#            tf.close()"""
        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)
  #      with open(testPath, 'r') as tsfile:
   #         testArticles = json.load(tsfile)


        ltrarticles = []

#        for a in trainArticles.keys():
#            trarticles=torch.tensor([], device=torch.device('cuda'))

 #           for x in a:
#                trarticles = torch.cat( (trarticles, self.voc.s2v(x).float().to(torch.device('cuda'))), 0 )
 #               ltrarticles.append((trarticles, int(self.labels[a])))

        trarticles = [([torch.cat( (trarticles, self.voc.s2v(x).long().to(torch.device('cuda')) ),0) , trainArticles[a]], int(self.labels[a]) ) for a in trainArticles.keys()]
        #tsarticles=torch.tensor([], device=torch.device('cuda'))
        #tsarticles = [(map(lambda x: torch.cat((tsarticles, self.voc.s2v(x).long().to(torch.device('cuda'))),0), testArticles[a]), int(self.labels[a])) for a in testArticles.keys()]

#        tsarticles = [(map(lambda x: self.voc.s2v(x).long().to(torch.device('cuda')), testArticles[a]), int(self.labels[a])) for a in testArticles.keys()]
        #trarticles = torch.tensor([], device=torch.device('cuda'))
        #trarticles = [(torch.cat((trarticles, self.voc.s2v(x).long().to(torch.device('cuda')) ),0), int(self.labels[a])) for x in trainArticles[a]]
        #trarticles = map(lambda x: torch.cat((trarticles, self.voc.s2v(x).long().to(torch.device('cuda')) ), 0), int(self.labels[a]), 
        return ltrarticles
