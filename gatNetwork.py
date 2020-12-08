import torch
import torch.nn as nn
import torch.nn.functional as F
from gatLayer import GATLayer, GATLayerBatchEnabled
from torch.nn.utils.rnn import pad_sequence
##word embeddings
##sentence-wise gat layer
##doc wise gat layer
##classification head

class SentenceEncoder(nn.Module):

    def __init__(self, wFeat, vocab, edim, labels=2, slope=0.01): ##Add BERT pretrained word embeddings

        ##sRep: sentence representaiton, can be dependency tree(1) or random graph(0)
        ##wFeat: desired feature size of word/sentence representations
        ##edim: embedding dimension
        ##vocab: vocab size
        ##labels: no of possible classifications of sentences

        super(SentenceEncoder, self).__init__()
        self.wFeat = wFeat; self.edim = edim

        self.embedding = nn.Embedding(vocab, self.edim)
        self.sEncoder = GATLayer(self.edim, self.wFeat, slope)

        ## Add final linear layers for classification
        self.classifier = nn.Linear(self.wFeat, labels)



    def forward(self, inSen, adj=None, selfLink=0):


        ##Get word representations
        words = self.embedding(inSen)

        if inSen.size()[0]!=0:

            ##Get sentence representation and pool
            sentence, attention = self.sEncoder(words, adj=adj, selfLink=selfLink)

        else: return

        poolSentence, _ = torch.max(sentence, dim=0)
        label = F.softmax(self.classifier(poolSentence), dim=0)

        return poolSentence, attention, sentence, label

class SentenceEncoder_batch(nn.Module):

    def __init__(self, wFeat, vocab, edim, labels=2, slope=0.0, device = torch.device('cuda')): ##Add BERT pretrained word embeddings

        ##sRep: sentence representaiton, can be dependency tree(1) or random graph(0)
        ##wFeat: desired feature size of word/sentence representations
        ##edim: embedding dimension
        ##vocab: vocab size
        ##labels: no of possible classifications of sentences

        super(SentenceEncoder_batch, self).__init__()
        self.wFeat = wFeat; self.edim = edim
        self.pad = vocab-1
        self.embedding = nn.Embedding(vocab, self.edim)
        self.sEncoder = GATLayerBatchEnabled(self.edim, self.wFeat, slope)
        self.device = device
        ## Add final linear layers for classification
        self.classifier = nn.Linear(self.wFeat, labels)



    def forward(self, inSen, adj=None):

        ##Initialize adjacency matrices to fully connected
        ##(I'll try dependency tree after this runs successfully)

        ##Get word representations
        doc = pad_sequence(inSen, batch_first=True, padding_value=self.pad)
        adj = torch.ones((doc.size()[0], doc.size()[1], doc.size()[1], 1), device=self.device)

        for d in (doc==self.pad).nonzero():

            adj[d[0].item(), d[1].item(), :, :] = torch.zeros_like(adj[0, 0, :, :])
            adj[d[0].item(), :, d[1].item(),:] = -1*torch.ones_like(adj[0, :, 0,:])


        words = self.embedding(doc)

        if doc.size()[0]!=0:

            ##Get sentence representation and pool
            sentence, attention = self.sEncoder(words, adj)
        else: return


        poolSentence = torch.mean(sentence, dim=1)

        label = F.softmax(self.classifier(poolSentence), dim=0)

        ##Play with the order of these operations
        ##Ex:
        ##sentenceClassified = self.classifier(sentence)
        ##sentenceClassified = F.softmax(sentenceClassified, dim = 1)

        return poolSentence, attention, sentence, label



class DocumentEncoder(nn.Module):

    def __init__(self, inFeat, sFeat, labels=2, slope=0.01):

        ##inFeat: input feature size
        ##sFeat: desired feature size of sentence representations
        ##labels: no of possible classifications of documents/sentences


        super(DocumentEncoder, self).__init__()
        self.inFeat = inFeat
        self.sFeat = sFeat

        self.dEncoder = GATLayer(self.inFeat, self.sFeat, slope)

        ## Add final linear layers for classification
        self.classifier = nn.Linear(self.sFeat, labels)



    def forward(self, inDoc, adj=None, selfLink=0):

        ##Initialize adjacency matrices to fully connected

        ##Get sentence representation and pool
        document, attention = self.dEncoder(inDoc, adj=adj, selfLink=selfLink)

        poolDocument, _ = torch.max(document, dim=0)

        label = F.softmax(self.classifier(poolDocument), dim=0)

        return poolDocument, attention, document, label


class SentenceLevelClassifier(nn.Module):
    def __init__(self):
        super(SentenceLevelClassifier, self).__init__()

        self.layer = nn.Linear(2, 2)
#        self.layerb = nn.Linear(2, 2)
        self.nLinearity = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inAtt):
        label = F.softmax(self.nLinearity(self.layer(inAtt)), dim=1)
        return label

class Article2Graph(nn.Module):

    def __init__(self, wFeat, vocab, edim, poolDim, labels=2, slope=0.01): ##Add BERT pretrained word embeddings

        ##sRep: sentence representaiton, can be dependency tree(1) or random graph(0)
        ##wFeat: desired feature size of word/sentence representations
        ##edim: embedding dimension
        ##poolDim: adaptive pooling shape
        ##labels: no of possible classifications of sentences

        ##i'll implement random graph first, then take care of dependency tree.

        super(Article2Graph, self).__init__()

        self.wFeat = wFeat

        self.embedding = nn.Embedding(vocab, edim)
        ##Add Bert pretrained word representations

        self.sEncoder = GATLayer(edim, self.wFeat, slope)
        self.dEncoder = GATLayer(self.wFeat, self.wFeat, slope)
        ## Add final linear layers for classification
        self.classifier = nn.Linear(self.wFeat, labels)



    def forward(self, inDoc, adjs):

        ##Initialize adjacency matrices to fully connected
        ##(I'll try dependency tree after this runs successfully)

#        adj = None; ##These will be needed if I implement dependency tree representation or something

        ##adj is zero 

        ##Get word representations
        words = self.embedding(inDoc)
        print(words)
        ##Get sentence representation
        if words.size()[0]!=0:
            words, sattention = self.sEncoder(words, adjs[0])
        else: return

#        poolSentence = torch.mean(sentence, dim=0); #print('poolSentence: ', poolSentence) #self.pool(sentence)
#        label = F.softmax(self.classifier(poolSentence), dim=0)
        document, dattention = self.dEncoder(words, adjs[1])

        document = document+words

        docMean = torch.mean(document, dim=0)

        return docMean, sattention, dattention, label



#class PropagandaDetector(nn.Module):

#    def __init__(self, inFeat, wFeat, vocab, edim, sFeat, wPool, sPool, sRep=0, sLabel=2, dLabel=2, slope=0.01):

#        super(PropagandaDetector, self).__init__()
#        self.inFeat=inFeat; self.wFeat = wFeat; self.sFeat = sFeat; self.wPool = wPool; self.sPool = self.sPool

#        self.sentenceEncoder = SentenceGAT(wFeat, vocab, edim, wPool, sRep=sRep, labels=sLabel, slope=slope)
#        self.documentEncoder = DocumentGAT(wPool, sFeat, poolDim=sPool, labels=dLabels, slope=slope)

#        self.sentenceClassifier = nn.Linear(wPool, sLabel)
#        self.documentClassifier = nn.Linear(sPool, dLabel)


