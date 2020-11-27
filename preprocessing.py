import numpy as np
import csv
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import torch
import time
#from streamlit import caching
import itertools
def preprocessor(datasetTrain, datasetTest, vocDict={'<UNK>':0}, path='dataset'):
    ##Create txt files where each file is an article and each  line is a sentence
    ##Create vocab dictionary (and save)

    #i = 0
    nltk.download('punkt')
#O#    voc = Vocab(vocDict)
#O#    labels = {}

    articlesTrain = {}
    with open(datasetTrain, 'r') as data:
       # if i%100==1:
       #     st = time.time()

        print("Processing the training set\n")
        reader = csv.reader(data, delimiter='\t')
        i = 0

        #articles = [[[s+'\n' for s in sent_tokenize(a[0])], a[4]] for a in itertools.islice(reader, 10)]
        #articleLen = [str(len(a[0]))+'\n' for a in itertools.islice(reader, 10)]
        #articleID = [a[4]+'\n' for a in itertools.islice(]
        #articlesTrain
        for a in reader:
##            i+=1
            articlesTrain[a[4]] = sent_tokenize(a[0])
#            if i==10:
        with open('articlesTrain.json', 'w') as artfile:
            json.dump(articlesTrain, artfile)
##                return
#        print(len(list(reader)))
#O#        for a in reader:
            #print("len: ", len(reader))
#            caching.clear_cache()
#O#            i+=1
#O#            if i%100 == 1:
#O#                st = time.time()
#O#            voc.addWords(a[0])

#O#            sentences = sent_tokenize(a[0])
#O#            id = a[4]
#O#            labels[id] = a[-1]

#O#            f = open(path+'/train/'+id+'.txt', 'w')
#O#            sentences = [s+'\n' for s in sentences]
            
#O#            f.writelines(sentences)
            #print(*sentences, file=f)
#O#            f.close()
#O#            if i%100 == 0:
#O#                et = time.time()
#O#                print("\nArticle no. ", i , " processed.")
#O#                print("Time taken: ", et-st)


#            if i == 500:
#O#        with open('vocab.json', 'w') as vocfile:
#O#            json.dump(voc.w2i, vocfile)
#                return

#O#        voc = Vocab(vocDict)
        #np.save('vocab.npy', voc.w2i, allow_pickle=True) #; np.save('labels.npy', labels)

    print("Processing the test set\n")
    articlesTest={}
    with open(datasetTest) as data:
        reader = csv.reader(data, delimiter='\t')
        for a in reader:
            articlesTest[a[4]] = sent_tokenize(a[0])

    with open('articlesTest.json', 'w') as artfile:
        json.dump(articlesTest, artfile)
##
        #i = 0
        #articlesTest = [sent_tokenize(a[0]) for a in reader]

#O#        for a in reader:
#            caching.clear_cache()
#O#            i +=1
#O#            if i%100 == 1:
#O#                st = time.time()

#O#            sentences = sent_tokenize(a[0])
#O#            id = a[4]
#O#            labels[id] = a[-1]

#O#            f = open(path+'/test/'+id+'.txt', 'w')

#            print(*sentences, file=f)
#O#            sentences = [s+'\n' for s in sentences]
#O#            f.writelines(sentences)

#O#            f.close()

#O#            if i%100 == 0:
#O#                et = time.time()
#O#                print("Article no. ", i , " processed.\n")
#O#                print("Time taken: ", et-st)


#           if i%100 == 0: print("Article no. ", i , " processed.\n")
#O#        with open('labels.json', 'w') as labelfile:
#O#            json.dump(labels, labelfile)

    ## save voc dict
    #np.save('vocab.npy', voc.w2i)
    #np.save('labels.npy', labels, allow_pickle=True)

#    return voc, labels

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

    def s2v(self, text):
        words = word_tokenize(text)
        #words = [w.lower() for w in words if w.isalpha()]
        sentence = torch.tensor([], device=torch.device('cuda'))

#        if '<EOS>' in list(self.w2i.keys()):
#            eos = self.w2i['<EOS>']
#        else:
#            self.w2i['<EOS>'] = self.keysize
#            eos = self.w2i['<EOS>']

        for i, w in enumerate(words):
            if w.isalpha():
                w = w.lower()
                if w in list(self.w2i.keys()):
                    sentence = torch.cat((sentence, torch.tensor([self.w2i[w]], device=torch.device('cuda')).float()), 0)
                else:
                    sentence = torch.cat((sentence, torch.tensor([0], device=torch.device('cuda')).float()), 0)
#        sentence = torch.cat((sentence, torch.tensor([eos], device=torch.device('cuda')).float()), 0 )
        return sentence



