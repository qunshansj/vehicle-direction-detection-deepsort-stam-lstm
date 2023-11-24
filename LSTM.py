python
import torch
import torch.nn as nn
import torch.nn.functional as F

class STAM_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(STAM_LSTM, self).__init__()
        self.feature_extraction = FeatureExtraction(input_dim, hidden_dim)
        self.spatial_attention = SpatialAttention(hidden_dim)
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.encoder = Encoder(hidden_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.spatial_attention(x)
        x = x.unsqueeze(1).repeat(1, x.size(1), 1)
        encoder_out, (hn, cn) = self.encoder(x)
        context = self.temporal_attention(encoder_out)
        decoder_in = context.unsqueeze(1)
        return self.decoder(decoder_in)
