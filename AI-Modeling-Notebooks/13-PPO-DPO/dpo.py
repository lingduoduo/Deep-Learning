import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PreferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, inputs):
        return self.network(inputs).squeeze(-1)

    def get_log_prob(self, inputs):
        return self.forward(inputs)


class DPO:
    def __init__(self, policy_model, reference_model, beta=0.1, grad_clip=1.0):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.beta = beta
        self.grad_clip = grad_clip

        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    @staticmethod
    def _preference_margin(chosen_log_probs, rejected_log_probs):
        return chosen_log_probs - rejected_log_probs

    def _get_reference_log_probs(self, chosen_inputs, rejected_inputs):
        with torch.no_grad():
            ref_chosen = self.reference_model.get_log_prob(chosen_inputs)
            ref_rejected = self.reference_model.get_log_prob(rejected_inputs)
        return ref_chosen, ref_rejected

    def compute_loss(self, chosen_inputs, rejected_inputs):
        policy_chosen = self.policy_model.get_log_prob(chosen_inputs)
        policy_rejected = self.policy_model.get_log_prob(rejected_inputs)
        ref_chosen, ref_rejected = self._get_reference_log_probs(
            chosen_inputs, rejected_inputs
        )

        policy_margin = self._preference_margin(policy_chosen, policy_rejected)
        reference_margin = self._preference_margin(ref_chosen, ref_rejected)
        logits = self.beta * (policy_margin - reference_margin)
        loss = -F.logsigmoid(logits).mean()

        reward_chosen = self.beta * (policy_chosen - ref_chosen).detach()
        reward_rejected = self.beta * (policy_rejected - ref_rejected).detach()

        metrics = {
            "loss": loss.item(),
            "accuracy": (logits > 0).float().mean().item(),
            "chosen_reward": reward_chosen.mean().item(),
            "rejected_reward": reward_rejected.mean().item(),
            "reward_margin": (reward_chosen - reward_rejected).mean().item(),
            "policy_margin": policy_margin.mean().item(),
            "reference_margin": reference_margin.mean().item(),
        }
        return loss, metrics

    def update(self, chosen_inputs, rejected_inputs):
        loss, metrics = self.compute_loss(chosen_inputs, rejected_inputs)

        self.policy_model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), max_norm=self.grad_clip
        )
        self.policy_model.optimizer.step()

        return metrics


def generate_dummy_preference_data(batch_size=32, input_dim=8, noise_scale=0.25):
    base = torch.randn(batch_size, input_dim)
    preference_signal = torch.randn(batch_size, input_dim) * noise_scale
    chosen_inputs = base + preference_signal
    rejected_inputs = base - preference_signal
    return chosen_inputs, rejected_inputs


def run_demo(steps=5, batch_size=32, input_dim=8):
    torch.manual_seed(42)

    policy_model = PreferenceModel(input_dim)
    reference_model = copy.deepcopy(policy_model)
    trainer = DPO(policy_model, reference_model, beta=0.1)

    latest_metrics = None
    for step in range(steps):
        chosen_inputs, rejected_inputs = generate_dummy_preference_data(
            batch_size=batch_size,
            input_dim=input_dim,
        )
        latest_metrics = trainer.update(chosen_inputs, rejected_inputs)
        print(
            f"step={step + 1} "
            f"loss={latest_metrics['loss']:.4f} "
            f"acc={latest_metrics['accuracy']:.4f} "
            f"margin={latest_metrics['reward_margin']:.4f}"
        )

    return latest_metrics


if __name__ == "__main__":
    metrics = run_demo()
    print("DPO training finished!")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print(f"Chosen reward: {metrics['chosen_reward']:.4f}")
    print(f"Rejected reward: {metrics['rejected_reward']:.4f}")
