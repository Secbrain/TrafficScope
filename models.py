import torch
from d2l import torch as d2l
from torch import nn
import math
import torch.nn.functional as F

from metaconst import TRAFFIC_SCOPE, TRAFFIC_SCOPE_TEMPORAL, TRAFFIC_SCOPE_CONTEXTUAL


class TemporalEncoder(d2l.Encoder):
    def __init__(self, packet_len, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TemporalEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(packet_len, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                  norm_shape, ffn_num_input, ffn_num_hiddens,
                                                  num_heads, dropout, use_bias))
        self.attention_weights = [None] * len(self.blks)
        self.temporal_features = None
        self.relu = nn.ReLU()

    def forward(self, X, valid_lens, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.relu(self.embedding(X)) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        self.temporal_features = X
        return X


class ContextualEncoder(d2l.Encoder):
    def __init__(self, agg_scale_num, freqs_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(ContextualEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(freqs_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.segment_encoding = nn.Embedding(agg_scale_num, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 d2l.EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                                  norm_shape, ffn_num_input, ffn_num_hiddens,
                                                  num_heads, dropout, use_bias))
        self.attention_weights = [None] * len(self.blks)
        self.contextual_features = None
        self.relu = nn.ReLU()

    def forward(self, X, contextual_segments, *args):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.relu(self.embedding(X)) * math.sqrt(self.num_hiddens)) + \
            self.segment_encoding(contextual_segments)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, torch.ones(X.size(0), device=X.device) * X.size(1))
            self.attention_weights[i] = blk.attention.attention.attention_weights
        self.contextual_features = X
        return X


class FusionEncoder(nn.Module):
    def __init__(self, temporal_dim, contextual_dim, num_hiddens, num_heads,
                 norm_shape, ffn_num_input, ffn_num_hiddens, dropout):
        super(FusionEncoder, self).__init__()

        assert num_hiddens % num_heads == 0, 'num_hiddens should be divided by num_heads'

        self.num_heads = num_heads
        self.num_hiddens = num_hiddens
        self.depth = self.num_hiddens // self.num_heads

        self.WQ = nn.Linear(temporal_dim, num_hiddens)
        self.WK = nn.Linear(contextual_dim, num_hiddens)
        self.WV = nn.Linear(contextual_dim, num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.addnorm1 = d2l.AddNorm(norm_shape, dropout)
        self.ffn = d2l.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = d2l.AddNorm(norm_shape, dropout)

        self.attention_weights = None
        self.fusion_features = None

    def forward(self, temporal_feature, contextual_feature):
        batch_size = temporal_feature.shape[0]

        q = self.WQ(temporal_feature)  # [batch_size, time_seq_len, num_hiddens]
        k = self.WK(contextual_feature)  # [batch_size, time_scale_len, num_hiddens]
        v = self.WV(contextual_feature)  # [batch_size, time_scale_len, num_hiddens]

        # --> [batch_size, num_heads, time_seq_len, depth]
        Q = q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # --> [batch_size, num_heads, time_scale_len, depth]
        K = k.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # --> [batch_size, num_heads, time_scale_len, depth]
        V = v.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # --> [batch_size, num_heads, time_seq_len, time_scale_len]
        attention_weights = torch.einsum('bnid,bnjd->bnij', Q, K)
        attention_weights = attention_weights * math.sqrt(self.num_hiddens)

        attention_weights = F.softmax(attention_weights, dim=-1)
        out = torch.einsum('bnij,bnjd->bnid', self.dropout(attention_weights), V)
        # --> [batch_size, time_seq_len, num_hiddens]
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_hiddens)
        out = self.addnorm1(q, out)
        out = self.addnorm2(out, self.ffn(out))

        self.attention_weights = attention_weights
        self.fusion_features = out
        return out


class TrafficScope(nn.Module):
    def __init__(self, temporal_seq_len, packet_len,
                 freqs_size, agg_scale_num, agg_points_num,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScope, self).__init__()
        self.model_name = TRAFFIC_SCOPE
        self.temporal_encoder = TemporalEncoder(packet_len, packet_len, packet_len, packet_len,
                                                packet_len, (temporal_seq_len, packet_len),
                                                packet_len, packet_len * 2, num_heads, num_layers, dropout)
        self.contextual_encoder = ContextualEncoder(agg_scale_num, freqs_size, freqs_size, freqs_size, freqs_size,
                                                    freqs_size, (agg_scale_num * agg_points_num, freqs_size),
                                                    freqs_size, freqs_size * 2,
                                                    num_heads, num_layers, dropout)
        self.fusion_encoder = FusionEncoder(packet_len, freqs_size, packet_len, num_heads,
                                            (temporal_seq_len, packet_len),
                                            packet_len, packet_len * 2, dropout)
        self.fc = nn.Linear(temporal_seq_len * packet_len, num_classes)

    def forward(self, temporal_data, temporal_valid_len, contextual_data, contextual_segments):
        temporal_feature = self.temporal_encoder(temporal_data, temporal_valid_len)
        contextual_feature = self.contextual_encoder(contextual_data, contextual_segments)
        out = self.fusion_encoder(temporal_feature, contextual_feature)
        out = F.softmax(self.fc(torch.flatten(out, start_dim=1)), dim=-1)
        return out

    def get_temporal_attention_weights(self):
        """
        should only call after forward function
        :return attention_weights List[attention_weight (batch_size x num_heads x query_size x key_size)]
        """
        return self.temporal_encoder.attention_weights

    def get_temporal_features(self):
        """
        should only call after forward function
        :return temporal_features ndarray batch_size x session_len x num_hiddens(=packet_len)
        """
        return self.temporal_encoder.temporal_features

    def get_contextual_attention_weights(self):
        """
        should only call after forward function
        :return attention_weights List[attention_weight (batch_size x num_heads x query_size x key_size)]
        """
        return self.contextual_encoder.attention_weights

    def get_contextual_features(self):
        """
        should only call after forward function
        :return contextual_features ndarray batch_size x (agg_scale_num x agg_points_num) x num_hiddens(=freqs)
        """
        return self.contextual_encoder.contextual_features

    def get_fusion_attention_weights(self):
        """
        should only call after forward function
        :return attention_weights (batch_size x num_heads x query_size x key_size)]
        """
        return self.fusion_encoder.attention_weights

    def get_fusion_features(self):
        """
        should only call after forward function
        :return fusion_features ndarray batch_size x temporal_seq_len x num_hiddens(=packet_len)
        """
        return self.fusion_encoder.fusion_features


class TrafficScopeTemporal(nn.Module):
    def __init__(self, temporal_seq_len, packet_len,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScopeTemporal, self).__init__()
        self.model_name = TRAFFIC_SCOPE_TEMPORAL
        self.temporal_encoder = TemporalEncoder(packet_len, packet_len, packet_len, packet_len,
                                                packet_len, (temporal_seq_len, packet_len),
                                                packet_len, packet_len * 2, num_heads, num_layers, dropout)
        self.fc = nn.Linear(temporal_seq_len * packet_len, num_classes)

    def forward(self, temporal_data, temporal_valid_len):
        temporal_feature = self.temporal_encoder(temporal_data, temporal_valid_len)
        out = F.softmax(self.fc(torch.flatten(temporal_feature, start_dim=1)), dim=-1)

        return out

    def get_attention_weights(self):
        """
        should only call after forward function
        :return attention_weights List[attention_weight (batch_size x num_heads x query_size x key_size)]
        """
        return self.temporal_encoder.attention_weights

    def get_temporal_features(self):
        """
        should only call after forward function
        :return temporal_features ndarray batch_size x session_len x num_hiddens(=packet_len)
        """
        return self.temporal_encoder.temporal_features


class TrafficScopeContextual(nn.Module):
    def __init__(self, agg_scale_num, agg_points_num, freqs_size,
                 num_heads, num_layers, num_classes, dropout):
        super(TrafficScopeContextual, self).__init__()
        self.model_name = TRAFFIC_SCOPE_CONTEXTUAL
        self.contextual_encoder = ContextualEncoder(agg_scale_num, freqs_size, freqs_size, freqs_size, freqs_size,
                                                    freqs_size, (agg_scale_num * agg_points_num, freqs_size),
                                                    freqs_size, freqs_size * 2,
                                                    num_heads, num_layers, dropout)
        self.fc = nn.Linear(agg_scale_num * agg_points_num * freqs_size, num_classes)

    def forward(self, contextual_data, contextual_segments):
        contextual_feature = self.contextual_encoder(contextual_data, contextual_segments)
        out = F.softmax(self.fc(torch.flatten(contextual_feature, start_dim=1)), dim=-1)

        return out

    def get_attention_weights(self):
        """
        should only call after forward function
        :return attention_weights List[attention_weight (batch_size x num_heads x query_size x key_size)]
        """
        return self.contextual_encoder.attention_weights

    def get_contextual_features(self):
        """
        should only call after forward function
        :return contextual_features ndarray batch_size x (agg_scale_num x agg_points_num) x num_hiddens(=freqs)
        """
        return self.contextual_encoder.contextual_features


# if __name__ == '__main__':
#     temporal_encoder = TemporalEncoder(64, 64, 64, 64, 64, (64, 64), 64, 128, 8, 2, 0.5)
#     contextual_encoder = ContextualEncoder(3, 128, 128, 128, 128, 128, (384, 128), 128, 256, 8, 2, 0.5)
#     fusion_encoder = FusionEncoder(64, 128, 64, 8, (64, 64), 64, 128, 0.5)
#
#     temporal_feature = temporal_encoder(torch.ones((2, 64, 64), dtype=torch.float), torch.tensor([32, 64]))
#     print(f'{temporal_feature = }')
#     print(f'{temporal_feature.size() = }')
#     loss = 20000 * temporal_feature.sum()
#     loss.backward()
#     print(f'temporal encoder grads {temporal_encoder.embedding.weight.grad}')
#
#     contextual_feature = contextual_encoder(torch.ones((2, 384, 128), dtype=torch.float),
#                                             torch.tensor([0] * 128 + [1] * 128 + [2] * 128))
#     print(f'{contextual_feature = }')
#     print(f'{contextual_feature.size() = }')
#     loss = 20000 * contextual_feature.sum()
#     loss.backward()
#     print(f'contextual encoder grads {contextual_encoder.embedding.weight.grad}')
#
#     temporal_feature = temporal_encoder(torch.ones((2, 64, 64), dtype=torch.float), torch.tensor([32, 64]))
#     contextual_feature = contextual_encoder(torch.ones((2, 384, 128), dtype=torch.float),
#                                             torch.tensor([0] * 128 + [1] * 128 + [2] * 128))
#     out = fusion_encoder(temporal_feature, contextual_feature)
#     print(f'{out = }')
#     print(f'{out.size() = }')
#     loss = 20000 * out.sum()
#     loss.backward()
#     print(f'fusion encoder grads {fusion_encoder.WQ.weight.grad}')
#
#     model = TrafficScope(64, 64, 128, 3, 128, 8, 2, 6, 0.5)
#     preds = model(torch.ones((2, 64, 64), dtype=torch.float), torch.tensor([32, 64]),
#                   torch.ones((2, 384, 128), dtype=torch.float),
#                   torch.tensor([0] * 128 + [1] * 128 + [2] * 128)
#                   )
#
#     print(f'{preds = }')
#     print(f'{model.get_temporal_features().size()}')
#     print(f'{model.get_contextual_features().size()}')
#     print(f'{model.get_fusion_features().size()}')
#     print(f'temporal attention weights = {model.get_temporal_attention_weights()}')
#     print(f'contextual attention weights = {model.get_contextual_attention_weights()}')
#     print(f'fusion attention_weights = {model.get_fusion_attention_weights()}')
#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(preds, torch.tensor([0, 1]))
#     loss.backward()
#     print(f'TrafficScope grads {model.temporal_encoder.embedding.weight.grad}')
#
#     model = TrafficScopeTemporal(64, 64, 8, 2, 6, 0.5)
#     preds = model(torch.ones((2, 64, 64), dtype=torch.float), torch.tensor([32, 64]))
#     print(f'{preds = }')
#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(preds, torch.tensor([0, 1]))
#     loss.backward()
#     print(f'TrafficScopeTemporal grads {model.temporal_encoder.embedding.weight.grad}')
#
#     model = TrafficScopeContextual(3, 128, 128, 8, 2, 6, 0.5)
#     preds = model(torch.ones((2, 384, 128), dtype=torch.float),
#                   torch.tensor([0] * 128 + [1] * 128 + [2] * 128))
#     print(f'{preds = }')
#     loss_fn = nn.CrossEntropyLoss()
#     loss = loss_fn(preds, torch.tensor([0, 1]))
#     loss.backward()
#     print(f'TrafficScopeContextual grads {model.contextual_encoder.embedding.weight.grad}')
