import torch
import torch.nn as nn

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

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class TextSentiment_attn(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class, device):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.attn = Attention(embed_dim, 8, device)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.attn(embedded, embedded, embedded)
        return self.fc(embedded)
