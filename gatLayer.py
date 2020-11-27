import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):

    """ONE LAYER OF GRAPH ATTENTION NETWORK

       Implementation of the graph attentional layer described in "Graph Attention Networks"
       by Veličković et al. (2018)
       Implementation inspired by the code of "Do Sentence Interactions Matter ? Leveraging Sentence Level
       Representations for Fake News Classification" by Vahibav, Annasamy and Hovy (2019)

       This is the building block of the graph attention layer
       An entire GAT model would be built by stacking one or multiple of these
       This layer computes the hidden representations of all nodes and normalized attention coefficients

       The current version initializes all documents as a fully connected directional graph with edge scores
       all set to 1, including self edges."""

    ## This thing doesn't really work on batches. Figure out why

    def __init__(self, inFeat, outFeat, slope=0.1):
        ## inFeat: input feature size
        ## outFeat: desired output feature size
        ## slope: negative slope of leaky relu

        super(GATLayer, self).__init__()

        self.inFeat = inFeat
        self.outFeat = outFeat
        self.slope = slope

        ## W: learnable params for hidden representations of nodes
        ## aParams: learnable params for attention coefficients
        ## aNLinearity: attentional non linearity factor for higher representation power
        ## hNLinearity: node representation non linearity factor for higher representation power

        self.W = nn.Linear(self.inFeat, self.outFeat)
        self.aParams = nn.Linear(2*self.outFeat, 1)
        self.aNLinearity = nn.LeakyReLU(negative_slope=self.slope)
        self.hNLinearity = nn.LeakyReLU(negative_slope=self.slope)
        #self.pooling = nn.AdaptiveAvgPool2d(self.poolDim)

    def forward(self, doc, adj=None):

        h = self.W(doc)
        numS = h.size()[0]

        ##Prepare node-node pairs
        fch = torch.cat([h.repeat(1, numS).view(numS*numS, -1), h.repeat(numS, 1)], dim=1).view(numS, -1, self.outFeat*2)

        att = self.aParams(fch)
        ##att.shape : (noOfWords, noOfWords, 1)


        att = self.aNLinearity(att)
        if adj == None:
            adj = torch.ones_like(att)

        att = torch.where(adj==0, -9e15*torch.ones_like(att), att)

        ##Normalize attention scores for each node neighborhood (aka set nodes with links going into the node)
        att = F.softmax(att, dim=1)
        att = att.squeeze(dim=2)
        ##att.shape : (noOfWords, noOfWords)
        ##rows are node neighborhoods

        h = self.hNLinearity(torch.matmul(att, h))

        ##final: h:(NoOfWords, outsize), att: (noOfWords, noOfWords)

        return h, att

