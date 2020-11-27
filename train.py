import torch
import torch.nn as nn
import torch.nn.functional as F
from gatNetwork import *
from data_loader import DataLoader, DataLoaderv2
from random import shuffle
import time


def trainGATs(trainPath, vocabPath, labelPath, wFeat, sFeat, edim, epoch=1,sLabels=2, dLabels=2, sSlope=0.01, dSlope=0.01, load=0, modelPath=None):

    """ Load all sentences without checking doc
        Feed them one by one to a SentenceEncoder
        Outputs form one giant tensor of vectors where each vector is a sentence
         Feed slices of the tensor (separated wrt document lenghts) one by one to docEncoder """


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##Prepare data loader, train/test parts
    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    ##Prepare models, loss, optimizer
    ##Add here: load params if continuing training from a previous model
    SGAT = SentenceEncoder(wFeat, v, edim, labels=sLabels, slope=sSlope).to(device)
    DGAT = DocumentEncoder(wFeat, sFeat, labels=dLabels, slope=dSlope).to(device)
    docLoss = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(list(SGAT.parameters())+list(DGAT.parameters()), lr=0.01, momentum=0.2)


    if load > 0:
        checkpoint = torch.load(modelPath)

        SGAT.load_state_dict(checkpoint['sentence_model'])
        DGAT.load_state_dict(checkpoint['document_model'])

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

        prevEpoch = 0


    ##Training loop
    encSents = torch.tensor([], device=device)
    trainArticles = dl.readData(trainPath)
    lenArt = len(trainArticles)

    for i in range(prevEpoch, epoch):
        #trainArticles = dl.readData(trainPath)

        print('epoch: ', i)

        stt = time.time()
        shuffle(trainArticles)

        lossphi.append(0); losspe.append(0); tposphi.append(0); fposphi.append(0)
        tnegphi.append(0); fnegphi.append(0); tpos.append(0); fpos.append(0)
        tneg.append(0); fneg.append(0)

        ##Iterate over articles in dataset
        for n, a in enumerate(trainArticles):

            article, l = a

            if l>0:
                l=torch.tensor([1], dtype=torch.long, device=device) #.to('cuda')
            else: l=torch.tensor([0], dtype=torch.long, device=device) #.to('cuda')

            ##Iterate over sentences in article
            sentences = [SGAT(s) for s in article]  # map(lambda s: SGAT(s), article)

            for s in sentences:
                if s!=None:
                    spooled, satt, _, _ = s
                    encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0) #.to('cuda')


            ##Forward and backward the article
            if len(encSents)!=0:

                dpooled, datt, dunpooled, label = DGAT(encSents)

                loss = docLoss(label.unsqueeze(0), l)

                lossphi[-1] += loss.item(); losspe[-1] += loss.item()

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

                opt.zero_grad()
                loss.backward()
                opt.step()

                encSents = torch.tensor([], device=device) #.to('cuda')

            ##Debugging, ignore
            ent = time.time()
            if n%99==0 and n!=0:
                lossphi[-1] = lossphi[-1]/100
                print('seconds taken for 100 iterations: ', ent-stt)
                print('average loss per 100 iterations: ', lossphi[-1], 'accuracy per 100 iterations: ', tposphi[-1]+tnegphi[-1]/100)
                print('confusion matrix for the last 100 iteration: ')
                print(tnegphi[-1], '	', fnegphi[-1], '\n', fposphi[-1],'	', tposphi[-1])

                tnegphi.append(0); fnegphi.append(0)
                tposphi.append(0); fposphi.append(0)
                lossphi.append(0)
        remain = lenArt%99

        if remain != 0:
            tnegphi[-1] = tnegphi[-1]/remain; fnegphi[-1] = fnegphi[-1]/remain
            tposphi[-1] = tposphi[-1]/remain; fposphi[-1] = fposphi[-1]/remain
            lossphi[-1] = lossphi[-1]/remain
            print('seconds taken for 100 iterations: ', ent-stt)
            print('average loss per 100 iterations: ', lossphi[-1], 'accuracy per 100 iterations: ', tposphi[-1]+tnegphi[-1]/100)
            print('confusion matrix for the last 100 iteration: ')
            print(tnegphi[-1], '	', fnegphi[-1], '\n', fposphi[-1],'	', tposphi[-1])

        losspe[-1] = losspe[-1]/lenArt
        print('\n~loss per epoch: ', losspe[-1])

    torch.save({ 'epoch': epoch,
                 'sentence_model': SGAT.state_dict(),
                 'document_model': DGAT.state_dict(),
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
               }, 'dataset/modelArticleOnly.tar')
    return 1, 2, 3, 4




def trainGATv2(trainPath, testPath, vocabPath, labelPath, wFeat, sFeat, sPool, dPool, edim, epoch=1, sRep=0, sLabels=2, dLabels=2, sSlope=0.01, dSlope=0.01):

    """ Load all sentences without checking doc
        Feed them one by one to a SentenceEncoder
        Outputs form one giant tensor of vectors where each vector is a sentence
         Feed slices of the tensor (separated wrt document lenghts) one by one to docEncoder """


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##Prepare data loader, train/test parts
    dl = DataLoaderv2(vocabPath, labelPath)
    v = dl.voc.keysize
    trainArticles, _ = dl.readData(trainPath, testPath)
    #trainArticles = list(trainArticles)
    ##Prepare models, loss, optimizer
    ##Add here: load params if continuing training from a previous model
#    SGAT = SentenceEncoder(wFeat, v, edim, wFeat, sRep=sRep, labels=sLabels, slope=sSlope).to(device)
#    DGAT = DocumentEncoder(wFeat, sFeat, poolDim=dPool, labels=dLabels, slope=dSlope).to(device)
    A2G = Article2Graph(wFeat, v+1, edim, dPool)


    docLoss = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(A2G.parameters(), lr=0.01, momentum=0.2)

    ##Training loop

    encSents = torch.tensor([], device=device) #.to(device)

    ##Tracking training loss and accuracy
    trainloss = 0; trainacc = 0

    for i in range(0, epoch):
        print('epoch: ', i)
#        trainArticles, _ = dl.readData(trainPath, testPath)
        shuffle(trainArticles)
#        trainloss.append(0); trainacc.append(0)

        ##Iterate over articles in dataset
        tt=0
        #print(torch.cat((trainArticles[0][0], torch.tensor([0])),0))
        #return
        for n, a in enumerate(trainArticles):
            print(a)
#            return
            tt+=1
            article, l = a

            if l>0:
                l=torch.tensor([1], dtype=torch.long, device=device) #.to('cuda')
            else: l=torch.tensor([0], dtype=torch.long, device=device) #.to('cuda')

            ##Iterate over sentences in article
#            sentences = [SGAT(s) for s in article]  # map(lambda s: SGAT(s), article)
            agraph = A2G(article, None) # torch.ones((article.size()[0], article.size()[0],1)))
            print(agraph)
            return
            for s in sentences:
                if s!=None:
                    spooled, satt, sunpooled, _ = s
                    encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0) #.to('cuda')


            ##Forward and backward the article
            if len(encSents)!=0:

                dpooled, datt, dunpooled, label = DGAT(encSents)

                loss = docLoss(label.unsqueeze(0), l)

                trainloss += loss.item()
                if l.item() == label.argmax().item(): trainacc += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                encSents = torch.tensor([], device=device) #.to('cuda')

            ##Debugging, ignore
            if n==100:
                print('loss: ', trainloss/100, 'accuracy: ', trainacc/100)
                trainloss = 0; trainacc = 0
                break
            #if n==300:
            #   print('loss: ', trainloss, 'accuracy: ', trainacc)

#    torch.save({ 'epoch': epoch,
#                 'sentence_model': SGAT.state_dict,
#                 'document_model': DGAT.state_dict,
#                 'optimizer': opt.state_dict
#               }, 'dataset/testModel.tar')
    return 1, 2, 3, 4
