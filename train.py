import torch
import torch.nn as nn
import torch.nn.functional as F
from gatNetwork import *
from data_loader import DataLoader, DataLoaderv2
from random import shuffle
import time
from sklearn.metrics import f1_score
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.decomposition import PCA

def testGATs(testPath, vocabPath, labelPath, modelPath, wFeat, sFeat, edim, sLabels = 2, dLabels = 2, sSlope=0.01, dSlope=0.01, selfLink=0, slc = 0, useBert=0):


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Model tested: ', modelPath)
    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    SGAT = SentenceEncoder(wFeat, v, edim, labels=sLabels, slope=sSlope, useBert=useBert).to(device)
    DGAT = DocumentEncoder(wFeat, sFeat, labels=dLabels, slope=dSlope).to(device)

    docLoss = nn.CrossEntropyLoss().to(device)

    checkpoint = torch.load(modelPath, map_location=device)

    SGAT.load_state_dict(checkpoint['sentence_model'])
    DGAT.load_state_dict(checkpoint['document_model'])
    SGAT.eval(); DGAT.eval()
    print("Epoch model is at: ", checkpoint['epoch'])
    avgloss = 0

    tpos = 0
    fpos = 0
    tneg = 0
    fneg = 0

    tposNorm = []
    fposNorm = []
    tnegNorm = []
    fnegNorm = []

    encSents = torch.tensor([], device=device)

    epoch = checkpoint['epoch']
    print('epoch:', epoch)

    if useBert==1:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)
        if slc == 0:
            testArticles = dl.readData_bert(testPath)
            sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, dl.noSample, replacement=False))
            lenArt = dl.lenArt
        else:
            testArticles = dl.readData_sentence(testPath, useBert)
            lenArt = dl.lenArt
    elif slc == 0:
        testArticles = dl.readData(testPath)
        lenArt = dl.lenArt #len(trainArticles)
    else:
        testArticles = dl.readData_sentence(testPath, useBert)
        lenArt = dl.lenArt

    stt = time.time()

    ##Iterate over articles in dataset

    predictions = []
    truelabels = []
    cosines = []; normdiffs = []
    plotpoints = torch.tensor([], device=device)
    plotpointsNorm = torch.tensor([], device=device)
    oProp = 0
    iProp = 0
    sentencelabels = []; normDifMulti = []

#    testArticles = testArticles[-2:-1]
    for n, a in enumerate(testArticles): # enumerate(trainArticles):
#        a = testArticles[-1]
        id, article, l = a

        if slc == 1:
            articleS = [s[0] for s in article]
            ls = [s[1] for s in article]
            article = articleS
            sentencelabels = sentencelabels+ls
            pca = PCA(n_components = 2)
        if useBert == 1:
            article = [bert(torch.tensor(tokenizer(s[:510] if len(s)>510 else s)['input_ids'], device=device).unsqueeze(0))[0].detach().squeeze(dim=0)[1:-1] for s in article]

        if l>0:
            l=torch.tensor([1], dtype=torch.long, device=device) #.to('cuda')
            truelabels.append(1)
        else:
            l=torch.tensor([0], dtype=torch.long, device=device) #.to('cuda')
            truelabels.append(0)
        ##Iterate over sentences in article
        sentences = [SGAT(s.to(device), selfLink=selfLink) for s in article]  # map(lambda s: SGAT(s), article)

        for s in sentences:
            if s!=None:
                spooled, satt, _, _ = s
                encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0)


        ##Forward the article
        if len( encSents)!=0:

            dpooled, datt, dunpooled, label = DGAT(encSents, selfLink=selfLink)
            if slc==1:
                cosine = torch.cosine_similarity(datt.float(), torch.transpose(datt, 0, 1).float(), dim=1)
                outNorm = datt.norm(dim=0); inNorm = datt.norm(dim=1)
                normDif = outNorm - inNorm #datt.norm(dim=0)-datt.norm(dim=1)

                datapoint  = torch.cat((cosine.unsqueeze(1), normDif.unsqueeze(1).float(), torch.tensor(ls, device=device).float().unsqueeze(1)), 1)
                plotpoints = torch.cat((plotpoints, datapoint), 0)

                datapointNorm = torch.cat((outNorm.unsqueeze(1).float(), inNorm.unsqueeze(1).float(), torch.tensor(ls, device=device).float().unsqueeze(1)), 1)
                plotpointsNorm = torch.cat((plotpointsNorm, datapointNorm), 0)

                outNormSort = torch.argsort(outNorm, descending=True); inNormSort = torch.argsort(inNorm, descending=False)

                if ls[outNormSort[0].item()] == 1:
                    oProp += 1
                if ls[inNormSort[0].item()] == 1:
                    iProp +=1


                sb.heatmap(datt.detach().to('cpu'), robust=True)
                plt.savefig('heatmaps/{modelName}-{aid}.png'.format(modelName=modelPath.replace('dataset/', "").replace('.tar', ''), aid = id))
                plt.clf()

            if l.item() == label.argmax().item():
                if l.item() == 0:
                    tneg += 1
                    predictions.append(0)
                else:
                    tpos += 1
                    predictions.append(1)
            else:
                if l.item() == 0:
                    fpos += 1
                    predictions.append(1)
                else:
                    fneg += 1
                    predictions.append(0)

            encSents = torch.tensor([], device=device)

    if slc==1:
        plotpoints = torch.transpose(plotpoints.to('cpu'), 0, 1).detach().numpy()
        plotPandas = pd.DataFrame({'cosine similarity': plotpoints[0], 'norm difference':plotpoints[1], 'z':plotpoints[2]})

        plotpointsNorm = torch.transpose(plotpointsNorm.to('cpu'), 0, 1).detach().numpy()
        plotPandasNorm = pd.DataFrame({'Norm of outgoing edge scores vector': plotpointsNorm[0], 'Norm of incoming edge scores vector': plotpointsNorm[1], 'z':plotpointsNorm[2]})

        sb.scatterplot(data=plotPandas, x='cosine similarity', y='norm difference', hue='z', palette='deep')
        plt.savefig('scatterplots/{modelName}-1trial.png'.format(modelName = modelPath.replace('dataset/', "").replace('.tar', '')))
        plt.clf()

        sb.scatterplot(data=plotPandasNorm, x='Norm of outgoing edge scores vector', y='Norm of incoming edge scores vector', hue='z', palette='deep')
        plt.savefig('normplots/{modelName}.png'.format(modelName = modelPath.replace('dataset/', "").replace('.tar', '')))


    ent = time.time()

    print('seconds taken for test: ', ent-stt)
    print('f1 score: ', f1_score(truelabels, predictions, average= None))
    print('confusion matrix of test: ')
    print(tneg, '	', fneg, '\n', fpos,'	', tpos)
    if slc==1:
        print('Percentage of highest outgoing edges that are propaganda: ', oProp/n)
        print('Percentage of highest incoming edges that are propaganda: ', iProp/n)
        print('Percentage of propaganda sentences in all sentences: ', sum(sentencelabels)/len(sentencelabels))

def trainGATs(trainPath, vocabPath, labelPath, wFeat, sFeat, edim, epoch=1,sLabels=2, dLabels=2, sSlope=0.01, dSlope=0.01, load=0, modelPath='dataset/readdatau5.tar', saveName='', debugMode=0, selfLink=0,  slcload=0, awise=0,useBert=0):

    """ Load all sentences without checking doc
        Feed them one by one to a SentenceEncoder
        Outputs form one giant tensor of vectors where each vector is a sentence
         Feed slices of the tensor (separated wrt document lenghts) one by one to docEncoder """



    device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##Prepare data loader, train/test parts
    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    ##Prepare models, loss, optimizer
    SGAT = SentenceEncoder(wFeat, v, edim, labels=sLabels, slope=sSlope, useBert=useBert).to(device)
    DGAT = DocumentEncoder(wFeat, sFeat, labels=dLabels, slope=dSlope).to(device)
#    SLC = SentenceLevelClassifier().to(device)

    docLoss = nn.CrossEntropyLoss().to(device) #weight = torch.tensor([0.1, 0.9])).to(device)
    #senLoss = nn.CrossEntropyLoss().to(device)
    senLoss = nn.CosineEmbeddingLoss()
    senLoss_verbose = nn.CosineEmbeddingLoss(reduction='none')
    senLoss2 = nn.L1Loss(reduction='none')

    if awise == 1:
        opt = torch.optim.Adam(list(SGAT.parameters())+list(DGAT.parameters()), lr= 0.001)
    else:
        opt = torch.optim.Adam(list(SGAT.parameters())+list(DGAT.parameters())+list(SLC.parameters()), lr=0.001)

    if load > 0:
        checkpoint = torch.load(modelPath, map_location=device)

        SGAT.load_state_dict(checkpoint['sentence_model'])
        DGAT.load_state_dict(checkpoint['document_model'])
        print(5)
        if slcload == 1:
            SLC.load_state_dict(checkpoint['slc_model'])

            opt.load_state_dict(checkpoint['optimizer'])
        else:
            if awise == 1:
                opt.load_state_dict(checkpoint['optimizer'])

        lossphi = checkpoint['lossphi']
        losspe = checkpoint['losspe']

        tposphi = checkpoint['tposphi']
        fposphi = checkpoint['fposphi']
        tnegphi = checkpoint['tnegphi']
        fnegphi = checkpoint['fnegphi']

        tpos = checkpoint['tpos']
        fpos = checkpoint['fpos']
        tneg = checkpoint['tneg']
        fneg = checkpoint['fneg']

        if slcload == 1:

            tposs = checkpoint['tposs']
            fposs = checkpoint['fposs']
            tnegs = checkpoint['tnegs']
            fnegs = checkpoint['fnegs']
        else:

            tposs = []
            fposs = []
            tnegs = []
            fnegs = []
            slosspe = 0
        prevEpoch = checkpoint['epoch']

    else:

        lossphi = []	##average loss per hundred doc iterations
        losspe = []	##average loss per epoch

        tposphi = []	##True positives per hundred doc iterations
        fposphi = []	##False positives per hundred doc iterations
        tnegphi = []	##etc...
        fnegphi = []

        tpos = []	##True positives per epoch
        fpos = []	##etc...
        tneg = []
        fneg = []

        tposs = []
        fposs = []
        tnegs = []
        fnegs = []

        prevEpoch = 0


    ##Training loop
    encSents = torch.tensor([], device=device)

    SGAT.train(); DGAT.train()
    if useBert==1:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)

    for i in range(prevEpoch, epoch):
        if useBert == 1:
            trainArticles = dl.readData_bert(trainPath)
            sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, dl.noSample, replacement=False))

        elif awise == 1:
            trainArticles = dl.readData_ub(trainPath)
            sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, dl.noSample, replacement=False))
        else:
            trainArticles = dl.readData_sentence(trainPath)
            shuffle(trainArticles)
            sample = list(range(0, len(trainArticles)))

        lenArt = dl.lenArt

        print('epoch: ', i)

        lossphi.append(0); losspe.append(0); tposphi.append(0); fposphi.append(0)
        tnegphi.append(0); fnegphi.append(0); tpos.append(0); fpos.append(0)
        tneg.append(0); fneg.append(0); tposs.append(0); fposs.append(0)
        tnegs.append(0); fnegs.append(0)
        stt = time.time()

        ##Iterate over articles in dataset
        for n, ind in enumerate(sample):

            a = trainArticles[ind]
            id, article, l = a

            if useBert == 1:

                article = [bert(torch.tensor(tokenizer(s[:510] if len(s)>510 else s)['input_ids']).to(device).unsqueeze(0))[0].detach().squeeze(dim=0)[1:-1] for s in article]

            if awise == 0:
                articleS = [s[0] for s in article]
                ls = [s[1] for s in article]
                article = articleS

            if l>0:
                l=torch.tensor([1], dtype=torch.long, device=device)
            else: l=torch.tensor([0], dtype=torch.long, device=device)

            ##Iterate over sentences in article:
            sentences = [SGAT(s.to(device), selfLink=selfLink) for s in article]

            for s in sentences:
                if s!=None:
                    spooled, satt, _, _ = s
                    encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0)

            ##Forward and backward the article
            if len(encSents)!=0:
#                print('encSents ', encSents.size())
                dpooled, datt, dunpooled, label = DGAT(encSents, selfLink=selfLink)
                loss = docLoss(label.unsqueeze(0), l)
                combinedLoss = loss
                ##Get sentence wise predictions
                if awise == 0:

                    #ls = torch.tensor(ls, dtype=torch.long, device=device)
                    #lsCosine = torch.where(ls==1, -1*torch.ones_like(ls, device=device), torch.ones_like(ls, device=device)).to(device)
                    #sloss = senLoss(datt.float(), torch.transpose(datt.float(), 0, 1), lsCosine.float())
                    #sloss_verbose = senLoss_verbose(datt.float(), torch.transpose(datt.float(), 0, 1), lsCosine.float())


                    ##Masked L1 loss
                    ls = torch.tensor(ls, dtype=torch.long, device=device)
                    lsMask = torch.where(ls==1, -1*torch.ones_like(ls, device=device), torch.ones_like(ls, device=device)).to(device)
                    sloss = senLoss2(datt.float(), torch.transpose(datt.float(), 0, 1)).mean(dim=1)
                    sloss = sloss*lsMask.float()
                    sloss = sloss.mean(dim=0)

                    combinedLoss = loss + sloss
                    #print("total loss: ", loss)
                    #return

                lossphi[-1] += loss.item(); losspe[-1] += loss.item()

                ##Create article wise confusion matrix stats
                if l.item() == label.argmax().item():
                    if l.item() == 0:
                        tnegphi[-1] += 1
                        tneg[-1] += 1
                    else:
                        tposphi[-1] += 1
                        tpos[-1] += 1
                else:
                    if l.item() == 0:
                        fposphi[-1] += 1
                        fpos[-1] += 1
                    else:
                        fnegphi[-1] += 1
                        fneg[-1] += 1
                ##Create sentence wise confusion matrix stats
                if awise == 0:
#                    confusion = torch.tensor(ls, dtype=torch.long, device=device) - 2*slabels.argmax(dim=1)

#                    tnegs[-1] += torch.sum(confusion == 0).item()
#                    tposs[-1] += torch.sum(confusion == -1).item()
#                    fnegs[-1] += torch.sum(confusion == 1).item()
#                    fposs[-1] += torch.sum(confusion == -2).item()
                     slosspe += sloss.item()


                opt.zero_grad()
                loss.backward()
                opt.step()


                encSents = torch.tensor([], device=device)

            ##Debugging, ignore
            if n%100==0 and n!=0:
                ent = time.time()
                lossphi[-1] = lossphi[-1]/100
                print('seconds taken for 100 iterations: ', ent-stt)
                print('average loss per 100 iterations: ', lossphi[-1], 'accuracy per 100 iterations: ', (tposphi[-1]+tnegphi[-1])/100)
                print('confusion matrix for the last 100 iteration: ')
                print(tnegphi[-1], '	', fnegphi[-1], '\n', fposphi[-1],'	', tposphi[-1])
                #if debugMode == 1:
                stt = time.time()
                tnegphi.append(0); fnegphi.append(0)
                tposphi.append(0); fposphi.append(0)
                lossphi.append(0)
#                return
        remain = lenArt%99

        if remain != 0 and debugMode == 0:
            ent = time.time()
            tnegphi[-1] = tnegphi[-1]; fnegphi[-1] = fnegphi[-1]
            tposphi[-1] = tposphi[-1]; fposphi[-1] = fposphi[-1]
            lossphi[-1] = lossphi[-1]/remain
            print('seconds taken for 100 iterations: ', ent-stt)
            print('average loss per 100 iterations: ', lossphi[-1], 'accuracy per 100 iterations: ', (tposphi[-1]+tnegphi[-1])/100)
            print('confusion matrix for the last 100 iteration: ')
            print(tnegphi[-1], '	', fnegphi[-1], '\n', fposphi[-1],'	', tposphi[-1])

        if awise == 0:

            print('cos losses for sentence classification: ')
            print(slosspe)
            slosspe = 0

        losspe[-1] = losspe[-1]/lenArt
        print('\n~loss per epoch: ', losspe[-1])
    if awise == 0:
        epoch = prevEpoch

    if debugMode == 0:
        torch.save({ 'epoch': epoch,
                     'sentence_model': SGAT.state_dict(),
                     'document_model': DGAT.state_dict(),
#                     'slc_model': SLC.state_dict(),
                     'optimizer': opt.state_dict(),
                     'losspe': losspe,
                     'lossphi': lossphi,
                     'tnegphi': tnegphi,
                     'fnegphi': fnegphi,
                     'tposphi': tposphi,
                     'fposphi': fposphi,
                     'tpos': tpos,
                     'fpos': fpos,
                     'tneg': tneg,
                     'fneg': fneg


                   }, 'dataset/{}.tar'.format(saveName))





def gridSearch(trainPath, vocabPath, labelPath, wFeats, sFeats, edims,  sLabels=2, dLabels=2, sSlope=0.01, dSlope=0.01, debugMode=0, selfLink=1, useBert=0):

    """ Load all sentences without checking doc
        Feed them one by one to a SentenceEncoder
        Outputs form one giant tensor of vectors where each vector is a sentence
         Feed slices of the tensor (separated wrt document lenghts) one by one to docEncoder """



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##Prepare data loader, train/test parts
    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize


    tpos = []	##True positives per epoch
    fpos = []	##etc...
    tneg = []
    fneg = []
    fscores = []
    params = []
    for w in wFeats:
        for s in sFeats:
            for e in edims:
                params.append([w, s, e])

    ##Training loop
    encSents = torch.tensor([], device=device)
    if useBert == 1:
        trainArticles = dl.readData_bert(trainPath)
        sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, 3000, replacement=False))
    else:
        trainArticles = dl.readData_ub(trainPath)
        sample = list(torch.utils.data.WeightedRandomSampler(dl.indices, 3000, replacement=False))
    articlesTurnedToList = {}

    if useBert==1:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device)


    for ttt, i in enumerate(params):

        tttt = True
#        trainArticles = dl.readData_sentence(trainPath)
#        shuffle(trainArticles)
#        sample = list(range(0, len(trainArticles)))

        lenArt = dl.lenArt
        w, s, e = i

        print('Parameters being tried:\nEmbedding size: ', e, '\nWord feature size: ', w, '\nSentence feature size: ', s)

        ##Prepare models, loss, optimizer
        SGAT = SentenceEncoder(w, v, e, labels=sLabels, slope=sSlope, useBert=useBert).to(device)
        DGAT = DocumentEncoder(w, s, labels=dLabels, slope=dSlope).to(device)

        docLoss = nn.CrossEntropyLoss().to(device) #weight = torch.tensor([0.1, 0.9])).to(device)

        opt = torch.optim.Adam(list(SGAT.parameters())+list(DGAT.parameters()), lr= 0.001)


        tpos.append(0); fpos.append(0); tneg.append(0); fneg.append(0)

        stt = time.time()
        #truelabels = []; predictions = []
        ##Epoch
        for epoch in range(0, 1):
            predictions = []; truelabels = []


            ##Iterate over articles in dataset
            for n, ind in enumerate(sample):
                if ind in articlesTurnedToList.keys():
                    a = articlesTurnedToList[ind]

                else:
                    a = trainArticles[ind]
                    #articlesTurnedToList[ind] = list(a)
                    id, article, l = a
                    article = [s for s in article]
                    a = [id, article, l]
                    articlesTurnedToList[ind] = a



                id, article, l = a
                if useBert == 1:
                    #article = [bert(torch.tensor(tokenizer(s)['input_ids']).unsqueeze(0))[0].squeeze(dim=0)[1:-1] for s in article]
                    article = [bert(torch.tensor(tokenizer(s[:510] if len(s)>510 else s)['input_ids'], device=device).unsqueeze(0))[0].detach().squeeze(dim=0)[1:-1] for s in article]



                if l>0:
                    l=torch.tensor([1], dtype=torch.long, device=device)
                    if n > 1999: truelabels.append(1)
                else:
                    l=torch.tensor([0], dtype=torch.long, device=device)
                    if n > 1999: truelabels.append(0)

                ##Iterate over sentences in article
                sentences = [SGAT(s.to(device), selfLink=selfLink) for s in article]

                for s in sentences:
                    if s!=None:
                        spooled, satt, _, _ = s
                        encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0)

                ##Forward and backward the article
                if len(encSents)!=0:

                    dpooled, datt, dunpooled, label = DGAT(encSents, selfLink=selfLink)

                    ##Create article wise confusion matrix stats
                    if n > 1999 :
                        if l.item() == label.argmax().item():
                            if l.item() == 0:
                                tneg[-1] += 1
                                predictions.append(0)
                            else:
                                tpos[-1] += 1
                                predictions.append(1)
                        else:
                            if l.item() == 0:
                                fpos[-1] += 1
                                predictions.append(1)
                            else:
                                fneg[-1] += 1
                                predictions.append(0)
                    else:
                        loss = docLoss(label.unsqueeze(0), l)

                        opt.zero_grad()
                        loss.backward()
                        opt.step()


                    encSents = torch.tensor([], device=device)


            print(truelabels, predictions)
            fscores.append(f1_score(truelabels, predictions))

            #print('\n~f1 score: ', fscores[-1])
            print('confusion matrix: ')
            print(tneg[-1], '	', fneg[-1], '\n', fpos[-1],'	', tpos[-1])
            tneg.append(0); fneg.append(0); fpos.append(0); tpos.append(0)
        print('\nf1 score: ', fscores[-1])

    w, s, e = params[torch.argmax(torch.tensor(fscores)).item()]
    print('Best performing set of parameters with respect to f1 score:\n Embedding size: ', e, '\nWord feature size: ', w, '\nSentence feature size: ', s)





def printPropSentence(path, label, vocab='vocab.json', sentence=1):


    dl = DataLoader(vocab, label)
    if sentence == 0:
        testArticles = dl.readData(path)
        lenArt = dl.lenArt #len(trainArticles)
    else:
        testArticles = dl.readData_sentence(path)
        lenArt = dl.lenArt

    for n, a in enumerate(testArticles): # enumerate(trainArticles):
#        a = testArticles[-1]
        id, article, l = a
        print('----ARTICLE ID: ', id)
        if sentence == 1:
            articleS = [s[0] for s in article]
            ls = [s[1] for s in article]
            article = articleS
            indices = []
            for ii, line in enumerate(ls):
                if line==1:
                    indices.append(ii)
            print('Propaganda lines: ', indices)

def testSentenceBaseline(testPath, vocabPath, labelPath):


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    testArticles = dl.readData_sentence(testPath, 0)
    lenArt = dl.lenArt
    stt = time.time()

    ##Iterate over articles in dataset


    oProp = 0
    iProp = 0
    sentencelabels = []
#    testArticles = testArticles[-2:-1]
    for n, a in enumerate(testArticles): # enumerate(trainArticles):
#        a = testArticles[-1]
        id, article, l = a
        articleS = [s[0] for s in article]
        ls = [s[1] for s in article]
        article = articleS
        sentencelabels = sentencelabels+ls

        sentences = [s for s in article]  # map(lambda s: SGAT(s), article)

        if len(sentences)!=0:

            datt = F.softmax(torch.rand((len(sentences), len(sentences))), dim=0)

#            if slc==1:

            outNorm = datt.norm(dim=0); inNorm = datt.norm(dim=1)

            outNormSort = torch.argsort(outNorm, descending=True); inNormSort = torch.argsort(inNorm, descending=False)
            if ls[outNormSort[0].item()] == 1:
                oProp += 1

            if ls[inNormSort[0].item()] == 1:
                iProp +=1



#    ent = time.time()

    print('Percentage of highest outgoing edges that are propaganda: ', oProp/n)
    print('Percentage of highest incoming edges that are propaganda: ', iProp/n)
    print('Percentage of propaganda sentences in all sentences: ', sum(sentencelabels)/len(sentencelabels))
