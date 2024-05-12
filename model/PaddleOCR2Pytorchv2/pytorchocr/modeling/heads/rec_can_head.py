"""
This code is refer from:
https://github.com/LBH1024/CAN/models/can.py
https://github.com/LBH1024/CAN/models/counting.py
https://github.com/LBH1024/CAN/models/decoder.py
https://github.com/LBH1024/CAN/models/attention.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import math
'''
Counting Module
'''


class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(), nn.Linear(channel // reduction, channel), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class CountingDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.trans_layer = nn.Sequential(
            nn.Conv2d(
                self.in_channel,
                512,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False),
            nn.BatchNorm2d(512))

        self.channel_att = ChannelAtt(512, 16)

        self.pred_layer = nn.Sequential(
            nn.Conv2d(
                512, self.out_channel, kernel_size=1, bias=False),
            nn.Sigmoid())

    def forward(self, x, mask):
        b, _, h, w = x.shape
        x = self.trans_layer(x)
        x = self.channel_att(x)
        x = self.pred_layer(x)

        if mask is not None:
            x = x * mask
        x = x.view(b, self.out_channel, -1)
        x1 = torch.sum(x, dim=-1)

        return x1, x.view(b, self.out_channel, h, w)


'''
Attention Decoder
'''


class PositionEmbeddingSine(nn.Module):
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_d = paddle.expand(paddle.to_tensor(2), dim_t.shape)
        # dim_t = self.temperature**(2 * (dim_t / dim_d).astype('int64') /
        #                            self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = torch.unsqueeze(x_embed, 3) / dim_t
        pos_y = torch.unsqueeze(y_embed, 3) / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class AttDecoder(nn.Module):
    def __init__(self, ratio, is_train, input_size, hidden_size,
                 encoder_out_channel, dropout, dropout_ratio, word_num,
                 counting_decoder_out_channel, attention):
        super(AttDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_channel = encoder_out_channel
        self.attention_dim = attention['attention_dim']
        self.dropout_prob = dropout
        self.ratio = ratio
        self.word_num = word_num

        self.counting_num = counting_decoder_out_channel
        self.is_train = is_train

        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        self.word_attention = Attention(hidden_size, attention['attention_dim'])

        self.encoder_feature_conv = nn.Conv2d(
            self.out_channel,
            self.attention_dim,
            kernel_size=attention['word_conv_kernel'],
            padding=attention['word_conv_kernel'] // 2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size,
                                               self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.counting_num,
                                                 self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num)

        if dropout:
            self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, cnn_features, labels, counting_preds, images_mask):
        if self.is_train:
            _, num_steps = labels.shape
        else:
            num_steps = 36

        batch_size, _, height, width = cnn_features.shape
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=cnn_features.device)
        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=cnn_features.device)

        hidden = self.init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        cnn_features_trans = self.encoder_feature_conv(cnn_features)

        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])

        cnn_features_trans = cnn_features_trans + pos

        word = torch.ones([batch_size]).long().to(device=cnn_features.device)  # init word as sos
        for i in range(num_steps):
            word_embedding = self.embedding(word)
            hidden = self.word_input_gru(word_embedding, hidden)
            word_context_vec, _, word_alpha_sum = self.word_attention(
                cnn_features, cnn_features_trans, hidden, word_alpha_sum,
                images_mask)

            current_state = self.word_state_weight(hidden)
            word_weighted_embedding = self.word_embedding_weight(word_embedding)
            word_context_weighted = self.word_context_weight(word_context_vec)

            if self.dropout_prob:
                word_out_state = self.dropout(
                    current_state + word_weighted_embedding +
                    word_context_weighted + counting_context_weighted)
            else:
                word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

            word_prob = self.word_convert(word_out_state)
            word_probs[:, i] = word_prob

            if self.is_train:
                word = labels[:, i]
            else:
                word = word_prob.argmax(1)
                word = torch.mul(
                    word, labels[:, i]
                )  # labels are oneslike tensor in infer/predict mode,  torch.multiply

        return word_probs

    def init_hidden(self, features, feature_mask):
        average = torch.sum(torch.sum(features * feature_mask, dim=-1),
                             dim=-1) / torch.sum(
                                 (torch.sum(feature_mask, dim=-1)), dim=-1)
        average = self.init_weight(average)
        return torch.tanh(average)


'''
Attention Module
'''


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        super(Attention, self).__init__()
        self.hidden = hidden_size
        self.attention_dim = attention_dim
        self.hidden_weight = nn.Linear(self.hidden, self.attention_dim)
        self.attention_conv = nn.Conv2d(
            1, 512, kernel_size=11, padding=5, bias=False)
        self.attention_weight = nn.Linear(
            512, self.attention_dim, bias=False)
        self.alpha_convert = nn.Linear(self.attention_dim, 1)

    def forward(self,
                cnn_features,
                cnn_features_trans,
                hidden,
                alpha_sum,
                image_mask=None):
        query = self.hidden_weight(hidden)
        alpha_sum_trans = self.attention_conv(alpha_sum)
        coverage_alpha = self.attention_weight(alpha_sum_trans.permute(0, 2, 3, 1))
        alpha_score = torch.tanh(
            query[:, None, None, :] + coverage_alpha + cnn_features_trans.permute(0, 2, 3, 1)
        )
        energy = self.alpha_convert(alpha_score)
        energy = energy - energy.max()
        energy_exp = torch.exp(torch.squeeze(energy, -1))

        if image_mask is not None:
            energy_exp = energy_exp * torch.squeeze(image_mask, 1)
        alpha = energy_exp / (energy_exp.sum(-1).sum(-1)[:,None,None] + 1e-10)
        alpha_sum = torch.unsqueeze(alpha, 1) + alpha_sum
        context_vector = torch.sum(
            torch.sum((torch.unsqueeze(alpha, 1) * cnn_features), -1), -1)

        return context_vector, alpha, alpha_sum


class CANHead(nn.Module):
    def __init__(self, in_channel, out_channel, ratio, attdecoder, **kwargs):
        super(CANHead, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.counting_decoder1 = CountingDecoder(self.in_channel,
                                                 self.out_channel, 3)  # mscm
        self.counting_decoder2 = CountingDecoder(self.in_channel,
                                                 self.out_channel, 5)

        self.decoder = AttDecoder(ratio, **attdecoder)

        self.ratio = ratio

    def forward(self, inputs, targets=None):
        cnn_features, images_mask, labels = inputs

        counting_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        counting_preds1, _ = self.counting_decoder1(cnn_features, counting_mask)
        counting_preds2, _ = self.counting_decoder2(cnn_features, counting_mask)
        counting_preds = (counting_preds1 + counting_preds2) / 2

        word_probs = self.decoder(cnn_features, labels, counting_preds,
                                  images_mask)
        return word_probs, counting_preds, counting_preds1, counting_preds2
