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
        logits = self.forward(inputs)
        return F.logsigmoid(logits)


class ORPO:
    def __init__(self, model, beta=0.1, odds_ratio_weight=1.0, grad_clip=1.0):
        self.model = model
        self.beta = beta
        self.odds_ratio_weight = odds_ratio_weight
        self.grad_clip = grad_clip

    @staticmethod
    def _log_odds(log_probs):
        probs = log_probs.exp().clamp(1e-8, 1 - 1e-8)
        return torch.logit(probs)

    def compute_loss(self, chosen_inputs, rejected_inputs):
        chosen_log_probs = self.model.get_log_prob(chosen_inputs)
        rejected_log_probs = self.model.get_log_prob(rejected_inputs)

        chosen_nll = -chosen_log_probs.mean()

        chosen_log_odds = self._log_odds(chosen_log_probs)
        rejected_log_odds = self._log_odds(rejected_log_probs)
        odds_ratio_logits = self.beta * (chosen_log_odds - rejected_log_odds)
        odds_ratio_loss = -F.logsigmoid(odds_ratio_logits).mean()

        total_loss = chosen_nll + self.odds_ratio_weight * odds_ratio_loss

        metrics = {
            "loss": total_loss.item(),
            "sft_loss": chosen_nll.item(),
            "or_loss": odds_ratio_loss.item(),
            "chosen_log_prob": chosen_log_probs.mean().item(),
            "rejected_log_prob": rejected_log_probs.mean().item(),
            "preference_accuracy": (odds_ratio_logits > 0).float().mean().item(),
        }
        return total_loss, metrics

    def update(self, chosen_inputs, rejected_inputs):
        loss, metrics = self.compute_loss(chosen_inputs, rejected_inputs)

        self.model.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.grad_clip
        )
        self.model.optimizer.step()

        return metrics


def generate_dummy_preference_data(batch_size=32, input_dim=8, noise_scale=0.4):
    base = torch.randn(batch_size, input_dim)
    preference_signal = torch.randn(batch_size, input_dim) * noise_scale
    chosen_inputs = base + preference_signal
    rejected_inputs = base - preference_signal
    return chosen_inputs, rejected_inputs


def run_demo(steps=5, batch_size=32, input_dim=8):
    torch.manual_seed(42)

    model = PreferenceModel(input_dim)
    trainer = ORPO(model, beta=0.1, odds_ratio_weight=1.0)

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
            f"sft={latest_metrics['sft_loss']:.4f} "
            f"or={latest_metrics['or_loss']:.4f} "
            f"acc={latest_metrics['preference_accuracy']:.4f}"
        )

    return latest_metrics


if __name__ == "__main__":
    metrics = run_demo()
    print("ORPO training finished!")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Final SFT loss: {metrics['sft_loss']:.4f}")
    print(f"Final OR loss: {metrics['or_loss']:.4f}")
    print(f"Preference accuracy: {metrics['preference_accuracy']:.4f}")
