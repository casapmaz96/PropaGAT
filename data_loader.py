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
        
        with open(vocPath, 'r') as vf:
            vocab = json.load(vf)
            self.voc = Vocab(vocab)

        with open(labelsPath, 'r') as lf:
            self.labels = json.load(lf) 
        self.change = None
        self.indices = None
        self.lenArt = 0
        self.noSample = 0

    def readData(self, trainPath):


        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)

        trarticles = [[a, map(lambda x: self.voc.s2v(x).long(), trainArticles[a]), int(self.labels[a])] for a in trainArticles.keys()]

        self.lenArt = len(trainArticles)

        return trarticles

    def readData_sentence(self, trainPath, bert=0):


        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)

        trarticles = []
        ap = 0
        co = 0
        for id in trainArticles.keys():

            if id in self.labels.keys():
                trarticles.append([id,[]])

                for i , s in enumerate(trainArticles[id]):
                    if len(s)>1:
                        co += 1
                        if bert==0:
                            sv = self.voc.s2v(s).long()
                            if sv.size()[0] > 0:
                                trarticles[-1][1].append((sv, self.labels[id][i]))
                                if self.labels[id][i] == 1:
                                    ap = 1

                        else:
                            sv = s
                            if len(sv) > 0:
                                trarticles[-1][1].append((sv, self.labels[id][i]))
                                if self.labels[id][i] == 1:
                                    ap = 1

                trarticles[-1].append(ap)

        self.lenArt = len(trarticles)

        return trarticles


    def readData_ub(self, trainPath):


        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)

        trarticles = [[a, map(lambda x: self.voc.s2v(x).long(), trainArticles[a]), int(self.labels[a])] for a in trainArticles.keys()]

        if self.change == None:
            for i, a in enumerate(trainArticles.keys()):
                if int(self.labels[a]) > 0:
                    self.change = i
                    break

            self.lenArt = len(trainArticles)
            indices = [1/(self.change*2)]*self.lenArt
            indices[i:] = [1/((self.lenArt-i)*2)]*(self.lenArt-i)
            self.indices = indices
            self.noSample = 2*(self.lenArt-i)

        return trarticles

    def readData_bert(self, trainPath):

        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)

        trarticles = [ [a, trainArticles[a], int(self.labels[a])] for a in trainArticles.keys()]

        if self.change == None:
            for i, a in enumerate(trainArticles.keys()):
                if int(self.labels[a]) > 0:
                    self.change = i
                    break

            self.lenArt = len(trainArticles)
            indices = [1/(self.change*2)]*self.lenArt
            indices[i:] = [1/((self.lenArt-i)*2)]*(self.lenArt-i)
            self.indices = indices
            self.noSample = 2*(self.lenArt-i)

        return trarticles


    def readData_with_padding(self, trainPath):

        self.voc.w2i['<PAD>'] = self.voc.keysize
        self.voc.keysize += 1

        with open(trainPath, 'r') as trfile:
            trainArticles = json.load(trfile)

        trarticles = [(map(lambda x: self.voc.s2v(x).long(), trainArticles[a]), int(self.labels[a])) for a in trainArticles.keys()]

        if self.change == None:
            for i, a in enumerate(trainArticles.keys()):
                if int(self.labels[a]) > 0:
                    self.change = i
                    break

            self.lenArt = len(trainArticles)
            indices = [1/(self.change*2)]*self.lenArt
            indices[i:] = [1/((self.lenArt-i)*2)]*(self.lenArt-i)
            self.indices = indices
            self.noSample = 2*(self.lenArt-i)

        return trarticles


