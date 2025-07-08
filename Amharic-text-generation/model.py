import torch
import torch.nn as nn

class YourModel(nn.Module):
    def __init__(self, config):
        super(YourModel, self).__init__()

        # Define your layers (example: embedding, linear layers, etc.)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.linear = nn.Linear(config.embedding_dim, config.num_classes)

        # Call the weight initialization function
        self.apply(self._init_weights)

    def forward(self, x):
        # Define the forward pass
        x = self.embedding(x)
        x = self.linear(x)
        return x

    def _init_weights(self, module):
        """Initialize weights for all layers in the model"""
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

# Configurations for the model (example)
class Config:
    vocab_size = 30522
    embedding_dim = 768
    num_classes = 10

# Example usage
config = Config()
model = YourModel(config)

# Optionally: print model to verify the structure
print(model)
