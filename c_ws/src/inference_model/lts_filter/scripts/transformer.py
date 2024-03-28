import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    """
    Input Embedding layer which consists of 2 stacked 1D convolutional layers with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

class SPCTReg(nn.Module):
    def __init__(self):
        super(SPCTReg, self).__init__()

        self.embedding = Embedding(3, 128)

        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)

        self.linear1 = nn.Sequential(
            nn.Conv1d(512, 2048, kernel_size=1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.linear2 = nn.Sequential(
            nn.Conv1d(2048 * 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.linear3 = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.convs = nn.Conv1d(256, 1, 1)

        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear1(x)

        batch_size, _, N = x.size()

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature], dim=1)  # 2048 * 3

        x = self.linear2(x)
        x = self.linear3(x)

        x = self.convs(x)

        x = self.sigmoid(x) 

        return x

