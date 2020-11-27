import torch
import torch.nn as nn

class CNN(nn.Module):
# 这里是为了得到风格矩阵T，进行sF和cF融合过程中的3层卷积的模块
    def __init__(self,layer,matrixSize=16):
        super(CNN,self).__init__()
        if(layer == 'r31'): #（当inputimg大小为batch x 3 x 512 x 512）输入 batch x 128 x 64 x 64
            self.convs = nn.Sequential(nn.Conv2d(64,32,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32,16,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(16,matrixSize,3,1,1))
                                       # batch x matrixSize x 64 x 64 (不改变长宽，只改变通道数)

        elif(layer == 'r41'): #（当inputimg大小为batch x 3 x 512 x 512）输入 batch x 64 x 128 x 128
            self.convs = nn.Sequential(nn.Conv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,32,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(32,matrixSize,3,1,1))
                                       # batch x matrixSize x 128 x 128 (不改变长宽，只改变通道数)

    def forward(self,x):
        out = self.convs(x)
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        return out

class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=16):
        super(MulLayer,self).__init__()
        self.snet = CNN(layer,matrixSize)
        self.cnet = CNN(layer,matrixSize)
        self.matrixSize = matrixSize
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        # 下面的compress操作保证了无论是用r31还是r41层来迁移，都能先将其统一为matrixSize大小的特征矩阵，从而能够同迁移矩阵T进行相乘
        if(layer == 'r41'):
            self.compress = nn.Conv2d(128,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,128,1,1,0)
        elif(layer == 'r31'):
            self.compress = nn.Conv2d(64,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,64,1,1,0)

        self.transmatrix = None

    def forward(self,cF,sF,trans=True): # cF对应为所挑选的需要转换的层的特征图张量
        cFBK = cF.clone()
        # print('-----下面为内容特征图尺寸-----')
        # print(cF.size())  # 打印出当前内容特征图的输出的尺寸大小
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1) # 将原大小为batch x channel x h x w的矩阵重新整理为batch x channel x hw大小的矩阵
        cMean = torch.mean(cFF,dim=2,keepdim=True) # dim=2表示在hw维度上进行均值计算
        cMean = cMean.unsqueeze(3) # 三个通道，逐个通道进行减去均值操作
        cMean = cMean.expand_as(cF) # 拓展成和上面cF同样的维度，便于直接相减
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS


        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)
        # compress_content batch x 32 x 4096(输入的feature map 的hw = 64 x 64 = 4096)

        if(trans):
            cMatrix = self.cnet(cF) #传输模块对应的3个卷积层执行结果
            sMatrix = self.snet(sF)
            transmatrix = torch.bmm(sMatrix, cMatrix.transpose(1,2))
            transmatrix = transmatrix.view(transmatrix.size(0),-1)
            transmatrix = self.fc(transmatrix)
            transmatrix = transmatrix.view(transmatrix.size(0),self.matrixSize,self.matrixSize)

            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out
