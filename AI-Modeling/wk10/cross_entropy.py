import torch

def lost_func(x, logits):
    '''
    Compute cross-entropy loss between predictions and true labels.
    Args:
        x: Tensor of shape (batch_size, seq_len) containing true labels.
        logits: Tensor of shape (batch_size, seq_len, vocab_size) containing raw model predictions. 
    Returns:
        loss: Scalar tensor representing the average cross-entropy loss.
    '''
    # Shift for next-token prediction (causal LM)
    shifted_logits = logits[:, :-1, :]     # (batch, seq_len-1, vocab_size)
    shifted_labels = x[:, 1:]              # (batch, seq_len-1)

    # Flatten for PyTorch CrossEntropyLoss -> (N, C) & (N)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1)
    )
    return loss

# Test the function
batch_size = 8
seq_len = 20
vocab_size = 100

# Random labels
x = torch.randint(0, vocab_size, (batch_size, seq_len))
print(x)

# Random raw logits (no softmax!)
logits = torch.randn(size=(batch_size, seq_len, vocab_size))
print(logits)

loss = lost_func(x, logits)
print("Loss:", loss)   
