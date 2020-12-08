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

def testGATs(testPath, vocabPath, labelPath, modelPath, wFeat, sFeat, edim, sLabels = 2, dLabels = 2, sSlope=0.01, dSlope=0.01, selfLink=0, slc = 0):


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    SGAT = SentenceEncoder(wFeat, v, edim, labels=sLabels, slope=sSlope).to(device)
    DGAT = DocumentEncoder(wFeat, sFeat, labels=dLabels, slope=dSlope).to(device)

    docLoss = nn.CrossEntropyLoss().to(device)

    checkpoint = torch.load(modelPath, map_location=device)

    SGAT.load_state_dict(checkpoint['sentence_model'])
    DGAT.load_state_dict(checkpoint['document_model'])
    SGAT.eval(); DGAT.eval()

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

    if slc == 0:
        testArticles = dl.readData(testPath)
        lenArt = dl.lenArt #len(trainArticles)
    else:
        testArticles = dl.readData_sentence(testPath)
        lenArt = dl.lenArt

    stt = time.time()

    ##Iterate over articles in dataset
    shuffle(testArticles)

    predictions = []
    truelabels = []
    for n, a in enumerate(testArticles): # enumerate(trainArticles):

        id, article, l = a

        if slc == 1:
            articleS = [s[0] for s in article]
            ls = [s[1] for s in article]
            article = articleS


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
        if len(encSents)!=0:

            dpooled, datt, dunpooled, label = DGAT(encSents, selfLink=selfLink)


#            for si, c in enumerate(cosine):
#                print(c, sentenceLabels[si])


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

        ##Debugging, ignore
#            ent = time.time()
#        if n == 100:
#            break

    sb.heatmap(datt.detach().to('cpu'), robust=True)
    plt.savefig('heatmap.png')

    print('id of the heatmap article: ', id, '\nsentence wise true labels of the heatmap article: ', ls)

    ent = time.time()

    print('seconds taken for test: ', ent-stt)
    print('f1 score: ', f1_score(truelabels, predictions, average= None))
    print('confusion matrix of test: ')
    print(tneg, '	', fneg, '\n', fpos,'	', tpos)



def trainGATs(trainPath, vocabPath, labelPath, wFeat, sFeat, edim, epoch=1,sLabels=2, dLabels=2, sSlope=0.01, dSlope=0.01, load=0, modelPath='dataset/readdatau5.tar', debugMode=0, selfLink=0,  slcload=0, awise=0):

    """ Load all sentences without checking doc
        Feed them one by one to a SentenceEncoder
        Outputs form one giant tensor of vectors where each vector is a sentence
         Feed slices of the tensor (separated wrt document lenghts) one by one to docEncoder """



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##Prepare data loader, train/test parts
    dl = DataLoader(vocabPath, labelPath)
    v = dl.voc.keysize

    ##Prepare models, loss, optimizer
    SGAT = SentenceEncoder(wFeat, v, edim, labels=sLabels, slope=sSlope).to(device)
    DGAT = DocumentEncoder(wFeat, sFeat, labels=dLabels, slope=dSlope).to(device)
    SLC = SentenceLevelClassifier().to(device)

    docLoss = nn.CrossEntropyLoss().to(device) #weight = torch.tensor([0.1, 0.9])).to(device)
    #senLoss = nn.CrossEntropyLoss().to(device)
    senLoss = nn.CosineEmbeddingLoss()
    senLoss_verbose = nn.CosineEmbeddingLoss(reduction='none')

    if awise == 1:
        opt = torch.optim.Adam(list(SGAT.parameters())+list(DGAT.parameters()), lr= 0.001)
    else:
        opt = torch.optim.Adam(list(SGAT.parameters())+list(DGAT.parameters())+list(SLC.parameters()), lr=0.001)

    if load > 0:
        checkpoint = torch.load(modelPath)

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
            slossV = []
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

    SGAT.train(); DGAT.train(); SLC.train()

    for i in range(prevEpoch, epoch):

        if awise == 1:
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

            if awise == 0:
                articleS = [s[0] for s in article]
                ls = [s[1] for s in article]
                article = articleS

            if l>0:
                l=torch.tensor([1], dtype=torch.long, device=device)
            else: l=torch.tensor([0], dtype=torch.long, device=device)

            ##Iterate over sentences in article
            sentences = [SGAT(s.to(device), selfLink=selfLink) for s in article]

            for s in sentences:
                if s!=None:
                    spooled, satt, _, _ = s
                    encSents = torch.cat((encSents, spooled.unsqueeze(0)), 0)

                    ##print("satt: ", satt)
                    ##sb.heatmap(satt, center=0)
                    ##plt.show()

            ##Forward and backward the article
            if len(encSents)!=0:
#                print('encSents ', encSents.size())
                dpooled, datt, dunpooled, label = DGAT(encSents, selfLink=selfLink)
                loss = docLoss(label.unsqueeze(0), l)

                ##Get sentence wise predictions
                if awise == 0:
                    #print(torch.mm(datt, datt)*torch.eye(datt.size()[0]))
                    #print(torch.sum(torch.mm(datt, datt)*torch.eye(datt.size()[0]), dim=1))

                    #mm = torch.sum(torch.mm(datt, datt)*torch.eye(datt.size()[0], device=device), dim=1)/(datt.norm(dim=0)*datt.norm(dim=1))
#0                    mm = torch.cosine_similarity(datt.float(), torch.transpose(datt, 0, 1).float(), dim=1)
#0                    normDif = datt.norm(dim=0)-datt.norm(dim=1)
                    ##also try: cosineembeddingloss
#0                    slabels = SLC(torch.cat((mm.unsqueeze(1), normDif.unsqueeze(1)), 1))
                    #print("labels: ",slabels)
#0                    sloss = senLoss(slabels, torch.tensor(ls, dtype=torch.long, device=device))

                    ls = torch.tensor(ls, dtype=torch.long, device=device)
                    lsCosine = torch.where(ls==1, -1*torch.ones_like(ls, device=device), torch.ones_like(ls, device=device)).to(device)
                    sloss = senLoss(datt.float(), torch.transpose(datt.float(), 0, 1), lsCosine.float())
                    sloss_verbose = senLoss_verbose(datt.float(), torch.transpose(datt.float(), 0, 1), lsCosine.float())

                    loss = loss + sloss
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
                     slossV.append(sloss_verbose)


                opt.zero_grad()
                loss.backward()
                opt.step()


                encSents = torch.tensor([], device=device)

            ##Debugging, ignore
#            break
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
#            print('confusion matrix for sentence classification: ')
#            print(tnegs[-1], '	', fnegs[-1], '\n', fposs[-1],'	', tposs[-1])
            print('cos losses for sentence classification: ')
            print(slossV)
            slossV = []

        losspe[-1] = losspe[-1]/lenArt
        print('\n~loss per epoch: ', losspe[-1])


    if debugMode == 0:
        torch.save({ 'epoch': epoch,
                     'sentence_model': SGAT.state_dict(),
                     'document_model': DGAT.state_dict(),
                     'slc_model': SLC.state_dict(),
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
#                     'tposs': tposs,
#                     'fposs': fposs,
#                     'tnegs': tnegs,
#                     'fnegs': fnegs

                   }, 'dataset/yesSelfLinkMax3.tar')





