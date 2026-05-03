import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """
    Direct Preference Optimization loss.

    Args:
        policy_chosen_logps: Log-probabilities of preferred responses
                             under the current policy.
        policy_rejected_logps: Log-probabilities of rejected responses
                               under the current policy.
        ref_chosen_logps: Log-probabilities of preferred responses
                          under the reference model.
        ref_rejected_logps: Log-probabilities of rejected responses
                            under the reference model.
        beta: Controls how strongly the model follows preference differences.

    Returns:
        Scalar DPO loss.
    """

    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    ref_log_ratio = ref_chosen_logps - ref_rejected_logps

    logits = beta * (policy_log_ratio - ref_log_ratio)

    loss = -F.logsigmoid(logits).mean()

    return loss


def sequence_log_probs(logits, labels, mask=None):
    """
    Compute sequence-level log-probabilities.

    Args:
        logits: Tensor, shape (batch_size, seq_len, vocab_size)
        labels: Tensor, shape (batch_size, seq_len)
        mask: Optional Tensor, shape (batch_size, seq_len)
              1 for valid tokens, 0 for padding tokens.

    Returns:
        Sequence log-probabilities, shape (batch_size,)
    """

    log_probs = F.log_softmax(logits, dim=-1)

    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    if mask is not None:
        token_log_probs = token_log_probs * mask
        return token_log_probs.sum(dim=-1)

    return token_log_probs.sum(dim=-1)

class DPOTrainer:
    def __init__(
        self,
        policy_model,
        reference_model,
        optimizer,
        beta=0.1,
        grad_clip=1.0,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.optimizer = optimizer
        self.beta = beta
        self.grad_clip = grad_clip

        # Reference model should be frozen
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_loss(self, batch):
        """
        batch should contain:
            chosen_input_ids
            chosen_labels
            chosen_mask
            rejected_input_ids
            rejected_labels
            rejected_mask
        """

        chosen_logits = self.policy_model(batch["chosen_input_ids"])
        rejected_logits = self.policy_model(batch["rejected_input_ids"])

        policy_chosen_logps = sequence_log_probs(
            chosen_logits,
            batch["chosen_labels"],
            batch["chosen_mask"],
        )

        policy_rejected_logps = sequence_log_probs(
            rejected_logits,
            batch["rejected_labels"],
            batch["rejected_mask"],
        )

        with torch.no_grad():
            ref_chosen_logits = self.reference_model(batch["chosen_input_ids"])
            ref_rejected_logits = self.reference_model(batch["rejected_input_ids"])

            ref_chosen_logps = sequence_log_probs(
                ref_chosen_logits,
                batch["chosen_labels"],
                batch["chosen_mask"],
            )

            ref_rejected_logps = sequence_log_probs(
                ref_rejected_logits,
                batch["rejected_labels"],
                batch["rejected_mask"],
            )

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=self.beta,
        )

        return loss

    def update(self, batch):
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.grad_clip,
        )

        self.optimizer.step()

        return loss.item()

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        logits = self.lm_head(x)
        return logits

def generate_dummy_dpo_batch(
    batch_size=4,
    seq_len=8,
    vocab_size=100,
):
    chosen_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    rejected_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    chosen_labels = chosen_input_ids.clone()
    rejected_labels = rejected_input_ids.clone()

    chosen_mask = torch.ones(batch_size, seq_len)
    rejected_mask = torch.ones(batch_size, seq_len)

    return {
        "chosen_input_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "chosen_mask": chosen_mask,
        "rejected_input_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
        "rejected_mask": rejected_mask,
    }

if __name__ == "__main__":
    vocab_size = 100

    policy_model = TinyLanguageModel(vocab_size)
    reference_model = TinyLanguageModel(vocab_size)

    # Usually, reference model starts as a copy of the initial policy model
    reference_model.load_state_dict(policy_model.state_dict())

    optimizer = optim.Adam(policy_model.parameters(), lr=1e-3)

    trainer = DPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        optimizer=optimizer,
        beta=0.1,
    )

    batch = generate_dummy_dpo_batch(vocab_size=vocab_size)

    loss = trainer.update(batch)

    print("DPO update completed!")
    print(f"Loss: {loss:.4f}")