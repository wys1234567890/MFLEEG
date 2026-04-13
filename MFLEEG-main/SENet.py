import torch
import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, nChan, reduction=16):
        """
        SENet模块，用于学习EEG通道之间的关系。
        专为 EEG 修改的 SENet
        Args:
            nChan: 输入的通道数（EEG通道数）。
            reduction: 通道压缩的比例，默认为16。
        """
        super(SENet, self).__init__()
        self.nChan = nChan

        # 全局平均池化：把 b 和 d 维度池化掉，只保留 c 通道
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 注意力层（输入通道数 = EEG 电极数）
        self.fc1 = nn.Linear(nChan, nChan // reduction, bias=False)  # 压缩通道
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(nChan // reduction, nChan, bias=False)  # 恢复通道
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #  把 c 换到第2位变成通道
        x = x.permute(0, 2, 1, 3)

        b, c, _, _ = x.size()  # 获取输入的形状 (batch_size, channels, height, width)
        y = self.global_avg_pool(x).view(b, c)  # 全局平均池化，输出形状 (batch_size, channels)
        y = self.fc1(y)  # 压缩通道
        y = self.relu(y)
        y = self.fc2(y)  # 恢复通道
        y = self.sigmoid(y).view(b, c, 1, 1)  # 生成通道权重，形状为 (batch_size, channels, 1, 1)

        x = x * y
        x = x.permute(0, 2, 1, 3)
        return x # 对输入的每个通道加权