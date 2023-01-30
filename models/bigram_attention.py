import sys
import os

sys.path.append(os.path.abspath(os.path.join('../')))
os.chdir('../')

import torch
import torch.nn as nn
from torch.nn import functional as F

from processing import read_txt_dataset
from processing import Vocab

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-3
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 200
embedding_size = 384
num_heads = 6
num_layers = 6
dropout = 0.2


torch.manual_seed(1337)

text = read_txt_dataset()
vocab = Vocab(text)
vocab_size = vocab.size

data = torch.tensor(vocab.encode(text), dtype=torch.long, device=device)
n = int(0.9 * len(data))
train_data = data[:n]
valid_data = data[n:]


class Head(nn.Module):
    """
    One head of self attention Module
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C*-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('inf'))
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi Heads of self attention in parallel
    """
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projections = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projections(out)
        return out


class LayerNorm:

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        # parameters (trained with backpropagation)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        # calculate the forward pass
        xmean = x.mean(1, keepdim=True)  # batch mean
        xvar = x.var(1, keepdim=True)  # batch variance
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize the input variant
        self.out = self.gamma * xhat * self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class BatchNorm1D:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backpropagation)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update'
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize the input variant
        self.out = self.gamma * xhat * self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class FeedForward(nn.Module):

    def __init__(self, num_embeddings: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embeddings, 4 * num_embeddings),
            nn.ReLU(),
            nn.Linear(4 * num_embeddings, num_embeddings),  # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, num_embeddings, n_heads):
        super().__init__()
        head_size = num_embeddings // n_heads
        self.sa_head = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(embedding_size)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))  # with residual connections
        x = x + self.ffwd(self.ln2(x))  # with residual connections
        return x


class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits from a lookup table
        self.token_embeddings_lookup = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.blocks = nn.Sequential(*[Block(embedding_size, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embedding_size)  # final layer norm
        self.sa_head = MultiHeadAttention(4, embedding_size // 4)
        self.lm_head = nn.Linear(embedding_size, vocab_size)
        self.ffwd = FeedForward(embedding_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # ids and logits are both (B,T,C), tensor of integers
        token_embeddings = self.token_embeddings_lookup(idx)
        posit_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = posit_embeddings + token_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # pytorch expects (B,C,T) tensor
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block size tokens
            idx_cond = idx[:, -block_size]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to ge the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = TransformerLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def get_batch(split):
    dataset = train_data if split == 'train' else valid_data
    ix = torch.randint(len(dataset) - block_size, (block_size, ))
    x = torch.stack([dataset[i: i + block_size] for i in ix])
    y = torch.stack([dataset[i + 1: i+block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = dict()
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


for step in range(max_iters):
    # every once in a while evaluate
    if step % eval_interval == 0:
        L = estimate_loss()
        print(F"step {step}: train loss {L['train']:.4f}, valid loss {L['valid']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate loss
    gen_logits, gen_loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    gen_loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(vocab.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
