import torch
import torch.nn as nn


class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernelSize, poolSize, *args, **kwargs):
        # 定义通用卷积块函数，参数说明：
        # inF: 输入通道数 | outF: 输出通道数 | dropoutP: dropout概率
        # kernelSize: 卷积核尺寸 | poolSize: 池化核尺寸
        # *args/**kwargs: 可变参数，传递给Conv2dWithConstraint
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(
                inF, outF, kernelSize, bias=False, max_norm=2, *args, **kwargs
            ),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, poolSize, *args, **kwargs):
        # 定义网络第一个卷积块（与通用块不同，多一层卷积）
        # 新增参数nChan: 输入数据的通道数（如脑电信号的电极数）
        return nn.Sequential(
            Conv2dWithConstraint(
                1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d(poolSize, stride=poolSize),
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        # 定义网络最后一个块，用于分类
        # outF: 输出通道数=类别数（如2分类则为2）
        return nn.Sequential(
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs),
            nn.LogSoftmax(dim=1),
        )

    def calculateOutSize(self, model, nChan, nTime):
        """
        # 计算经过指定网络层后的特征图尺寸，参数：
        # model: 待测试的网络层 | nChan: 输入通道数 | nTime: 时间维度长度（如脑电信号的时间点数量）
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, poolSize, localKernalSize):
        super().__init__()

        nClass = 2
        dropoutP = 0.5
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(
            nFilt_FirstLayer,
            dropoutP,
            localKernalSize["LocalLayers"][0],
            nChan,
            poolSize["LocalLayers"][0],
        )

        middleLayers = nn.Sequential(
            *[
                self.convBlock(inF, outF, dropoutP, kernalS, poolS)
                for inF, outF, kernalS, poolS in zip(
                    nFiltLaterLayer[:-1],
                    nFiltLaterLayer[1:-1],
                    localKernalSize["LocalLayers"][1:],
                    poolSize["LocalLayers"][1:],
                )
            ]
        )
        firstGlobalLayer = self.convBlock(
            nFiltLaterLayer[-2],
            nFiltLaterLayer[-1],
            dropoutP,
            localKernalSize["GlobalLayers"],
            poolSize["GlobalLayers"],
        )

        self.allButLastLayers = nn.Sequential(
            firstLayer, middleLayers, firstGlobalLayer
        )

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

        self.weight_keys = [
            [
                "allButLastLayers.0.0.weight",
                "allButLastLayers.0.0.bias",
                "allButLastLayers.0.1.weight",
            ],
            ["allButLastLayers.1.0.1.weight"],
            ["allButLastLayers.1.1.1.weight"],
            ["allButLastLayers.2.1.weight"],
            ["lastLayer.0.weight", "lastLayer.0.bias"],
        ]

    def forward(self, x):
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        """
        doWeightNorm：布尔值，是否开启权重归一化（默认 True）；
        max_norm：浮点数，权重的最大 L2 范数（默认 1）。
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)
