import torch
import torch.nn as nn
                     
class LukeConvBlock(nn.Module):
    def __init__(self, insize, outsize, batchnorm=True, cn_dropout=0):
        super().__init__()
        self.insize = insize
        self.outsize = outsize
        self.batchnorm = batchnorm
        block  = []
        block.append(nn.Conv2d(insize, outsize, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        if cn_dropout>0:
            block.append(nn.Dropout2d(cn_dropout))
        block.append(nn.Conv2d(outsize, outsize, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        if batchnorm:
            block.append(nn.BatchNorm2d(outsize))
        if cn_dropout>0:
            block.append(nn.Dropout2d(cn_dropout))
        block.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*block)
        
    def forward(self, x):
        return self.block(x)
        
                     
class Luke(nn.Module):    
    def __init__(self, fl_features=16, channels=1, depth=3, batchnorm=True, cn_dropout=0):
        super().__init__()
        
        cnn_seq = []
        cnn_seq.append(LukeConvBlock(channels,fl_features,batchnorm,cn_dropout=cn_dropout))
        for i in range(depth-1):
            cnn_seq.append(LukeConvBlock(fl_features*2**(i),fl_features*2**(i+1),batchnorm,cn_dropout=cn_dropout))
        
        cnn_seq[-1].block[-1] = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.cnn_seq = nn.Sequential(*cnn_seq)
        self.nfeat_afterconv = 7*7*fl_features*2**(i+1)
        classifier = []
        classifier.append(nn.Linear(self.nfeat_afterconv,2000))
        classifier.append(nn.ReLU())
        if batchnorm:
            classifier.append(nn.BatchNorm1d(2000))
        classifier.append(nn.Linear(2000,1000))
        classifier.append(nn.ReLU())
        if batchnorm:
            classifier.append(nn.BatchNorm1d(1000))
        classifier.append(nn.Linear(1000,4))
        self.classifier = nn.Sequential(*classifier)
        
    def forward(self, x):
        x = self.cnn_seq(x)
        x = x.view(-1,self.nfeat_afterconv)
        return self.classifier(x)