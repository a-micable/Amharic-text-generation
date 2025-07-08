import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32         # Number of sequences processed in parallel
block_size = 8          # Maximum context length for predictions
max_iterations = 3000   # Number of training iterations
eval_interval = 300     # Interval for evaluating the model
learning_rate = 1e-2     # Learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_iterations = 200  # Number of evaluation iterations

# Set seed for reproducibility
torch.manual_seed(1337)

# Load and preprocess text data
with open('am_data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Create a mapping from characters to integers
unique_chars = sorted(set(text))
vocab_size = len(unique_chars)
char_to_index = {char: idx for idx, char in enumerate(unique_chars)}
index_to_char = {idx: char for idx, char in enumerate(unique_chars)}

# Encoding and decoding functions
encode = lambda s: [char_to_index[char] for char in s]
decode = lambda indices: ''.join(index_to_char[idx] for idx in indices)

# Split data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

def get_batch(split):
    """Generate a batch of data for training or validation."""
    dataset = train_data if split == 'train' else val_data
    indices = torch.randint(len(dataset) - block_size, (batch_size,))
    x_batch = torch.stack([dataset[i:i + block_size] for i in indices])
    y_batch = torch.stack([dataset[i + 1:i + block_size + 1] for i in indices])
    return x_batch.to(device), y_batch.to(device)

@torch.no_grad()
def evaluate_loss():
    """Evaluate loss on training and validation sets."""
    model.eval()
    losses = {'train': torch.zeros(evaluation_iterations), 'val': torch.zeros(evaluation_iterations)}
    
    for split in ['train', 'val']:
        for i in range(evaluation_iterations):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[split][i] = loss.item()
    
    model.train()
    return {split: losses[split].mean() for split in ['train', 'val']}

class BigramLanguageModel(nn.Module):
    """Simple bigram language model."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.embedding_table(idx)
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate text from the model."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

# Initialize model and optimizer
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iteration in range(max_iterations):
    if iteration % eval_interval == 0:
        losses = evaluate_loss()
        print(f"Iteration {iteration}: Training Loss {losses['train']:.4f}, Validation Loss {losses['val']:.4f}")

    # Get a batch of data
    x_batch, y_batch = get_batch('train')

    # Compute loss and update model
    _, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text from the trained model
initial_context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text_indices = model.generate(initial_context, max_new_tokens=500)[0].tolist()
print(decode(generated_text_indices))
