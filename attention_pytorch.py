import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_feature, num_head, device, mask_right=False):
        super(Attention, self).__init__()
        self.in_feature = in_feature
        self.num_head = num_head
        self.size_per_head = in_feature // num_head
        self.out_dim = num_head * self.size_per_head
        assert self.size_per_head * num_head == in_feature, "in_feature must be divisible by num_head"
        self.mask_right = mask_right
        self.q_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.k_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.v_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.scale = torch.sqrt(torch.FloatTensor([self.size_per_head])).to(device)
        self.fc = nn.Linear(in_feature, in_feature)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = key.size(0)

        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        query = query.view(batch_size, self.num_head, -1, self.size_per_head)
        key = key.view(batch_size, self.num_head, -1, self.size_per_head)
        value = value.view(batch_size, self.num_head, -1, self.size_per_head)

        energy = torch.matmul(query, key.permute(0,1,3,2)) / self.scale

        if attn_mask is not None:
            energy = energy.masked_fill(attn_mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.in_feature)

        x = self.fc(x)

        return x.squeeze(1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, num_filters=3, filter_sizes=[2,3,4]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes]
        )
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text):
        embedded = self.embedding(text[0])
        embedded = embedded.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(embedded, conv) for conv in self.convs], 1)
        return self.fc(out)

class TextCNN_attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, device, num_filters=3, filter_sizes=[2,3,4]):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=False)
        # self.attn = MultiHeadAttention(in_features=embed_dim, head_num=8)
        self.attn = Attention(9, 3, device)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes]
        )
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text):
        embedded = self.embedding(text[0])
        embedded = embedded.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(embedded, conv) for conv in self.convs], 1)
        embedded = self.attn(out, out, out)
        return self.fc(out)
