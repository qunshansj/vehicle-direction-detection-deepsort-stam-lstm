python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, input_dim):
        super(SpatialAttention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.linear(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class TemporalAttention(nn.Module):
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        attention_weights = F.softmax(self.linear(x), dim=1)
        return torch.sum(attention_weights * x, dim=1)

class FeatureExtraction(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeatureExtraction, self).__init__()
        self.global_feature = nn.Linear(input_dim, hidden_dim)
        self.local_feature = nn.Linear(input_dim, hidden_dim)
        self.feature_fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        global_feature = self.global_feature(x)
        local_feature = self.local_feature(x)
        fusion_feature = torch.cat((global_feature, local_feature), dim=-1)
        return self.feature_fusion(fusion_feature)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        return self.lstm(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out)
