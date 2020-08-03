import torch
from attention_pytorch import *

import torchtext
from torchtext.datasets import text_classification
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label

import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', vocab=None)

train_len = int(len(train_dataset) * 0.95)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 128
NUN_CLASS = len(train_dataset.get_labels())

#model_name = "./.normal"
#model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

model_name = "./.attention"
model = TextSentiment_attn(VOCAB_SIZE, EMBED_DIM, NUN_CLASS, device).to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)


def train_func(sub_train_):
    # 训练模型
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc / len(text),
        }
        if not os.path.exists(model_name):
            os.makedirs(model_name)
        torch.save(state, model_name + '/model_' + str(i) +'.pth')
        if i % 100 == 0:
            print('step {}: loss: {}, acc: {}'.format(i, loss.item() / len(cls), (output.argmax(1) == cls).sum().item() / len(cls)))

    # 调整学习率
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)

if os.path.exists(model_name):
    checkpoint = torch.load(os.walk(model_name)[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    dev_best_loss = checkpoint['dev_best_loss']

train_func(sub_train_)