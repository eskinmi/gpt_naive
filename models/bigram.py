import torch
import torch.nn as nn
from torch.nn import functional as F

from processing import read_txt_dataset
from processing import Vocab


batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_size = 32


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
            losses[k] = loss.items()
        out[split] = losses.mean()
    model.train()
    return out


class BiGramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits from a lookup table
        self.token_embeddings_lookup = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(block_size, embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # ids and logits are both (B,T,C), tensor of integers
        token_embeddings = self.token_embeddings_lookup[idx]
        posit_embeddings = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = posit_embeddings + token_embeddings
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
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to ge the probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":

    torch.manual_seed(1337)

    text = read_txt_dataset()
    vocab = Vocab(text)
    vocab_size = vocab.size

    data = torch.tensor(vocab.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    valid_data = data[n:]

    model = BiGramLanguageModel()
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        # every once in a while evaluate
        if step % eval_interval == 0:
            L = estimate_loss()
            print(F"step {step}: train loss {L['train']:.4f}, valid loss {L['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate loss
        gen_logits, gen_loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        gen_loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(vocab.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
