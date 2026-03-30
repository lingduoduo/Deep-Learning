import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, states):
        return self.network(states)

    def get_log_prob(self, states, actions):
        logits = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions)


class GRPO:
    def __init__(
        self,
        policy,
        ref_model=None,
        clip_epsilon=0.2,
        beta=0.1,
        grad_clip=1.0,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.grad_clip = grad_clip

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    @staticmethod
    def _masked_mean(values, mask):
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (values * mask).sum(dim=1, keepdim=True) / denom

    def compute_group_advantage(self, rewards, mask):
        mask = mask.float()
        mean = self._masked_mean(rewards, mask)
        centered = (rewards - mean) * mask
        variance = self._masked_mean(centered.pow(2), mask)
        std = torch.sqrt(variance + 1e-8)
        advantages = centered / std
        return advantages.detach()

    def _get_reference_log_probs(self, states, actions, ref_log_probs=None):
        if ref_log_probs is not None:
            return ref_log_probs
        if self.ref_model is None:
            raise ValueError("ref_model or ref_log_probs must be provided for GRPO.")
        with torch.no_grad():
            return self.ref_model.get_log_prob(states, actions)

    def compute_loss(
        self,
        states,
        actions,
        old_log_probs,
        rewards,
        mask,
        ref_log_probs=None,
    ):
        mask = mask.float()
        batch_size, group_size = rewards.shape
        advantages = self.compute_group_advantage(rewards, mask).reshape(-1)
        flat_mask = mask.reshape(-1)
        current_log_probs = self.policy.get_log_prob(states, actions)
        reference_log_probs = self._get_reference_log_probs(
            states, actions, ref_log_probs=ref_log_probs
        )

        ratios = torch.exp(current_log_probs - old_log_probs)
        clipped_ratios = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        policy_objective = torch.min(
            ratios * advantages,
            clipped_ratios * advantages,
        )
        policy_loss = -policy_objective

        log_ratio = current_log_probs - reference_log_probs
        kl_penalty = torch.exp(log_ratio) - log_ratio - 1.0

        total_loss = policy_loss + self.beta * kl_penalty
        masked_loss = total_loss * flat_mask
        mean_loss = masked_loss.sum() / flat_mask.sum().clamp_min(1e-8)

        metrics = {
            "loss": mean_loss.item(),
            "group_size": group_size,
            "mean_advantage": (
                (advantages * flat_mask).sum() / flat_mask.sum().clamp_min(1e-8)
            ).item(),
            "mean_ratio": (
                (ratios * flat_mask).sum() / flat_mask.sum().clamp_min(1e-8)
            ).item(),
            "mean_kl": (
                (kl_penalty * flat_mask).sum() / flat_mask.sum().clamp_min(1e-8)
            ).item(),
        }
        return mean_loss, metrics

    def update(
        self,
        states,
        actions,
        old_log_probs,
        rewards,
        mask,
        ref_log_probs=None,
    ):
        loss, metrics = self.compute_loss(
            states,
            actions,
            old_log_probs,
            rewards,
            mask,
            ref_log_probs=ref_log_probs,
        )

        self.policy.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.grad_clip
        )
        self.policy.optimizer.step()

        return metrics


def generate_dummy_group_data(
    batch_size=8,
    group_size=4,
    state_dim=6,
    action_dim=3,
):
    states = torch.randn(batch_size, group_size, state_dim)
    logits = torch.randn(batch_size, group_size, action_dim)
    dist = torch.distributions.Categorical(logits=logits)
    actions = dist.sample()
    old_log_probs = dist.log_prob(actions)
    rewards = torch.randn(batch_size, group_size)
    mask = torch.ones(batch_size, group_size)

    flat_states = states.reshape(batch_size * group_size, state_dim)
    flat_actions = actions.reshape(batch_size * group_size)
    flat_old_log_probs = old_log_probs.reshape(batch_size * group_size)

    return flat_states, flat_actions, flat_old_log_probs, rewards, mask


def run_demo(steps=5, batch_size=8, group_size=4, state_dim=6, action_dim=3):
    torch.manual_seed(42)

    policy = Policy(state_dim, action_dim)
    ref_model = copy.deepcopy(policy)
    trainer = GRPO(policy, ref_model=ref_model, beta=0.1)

    latest_metrics = None
    for step in range(steps):
        states, actions, old_log_probs, rewards, mask = generate_dummy_group_data(
            batch_size=batch_size,
            group_size=group_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        flat_rewards = rewards.reshape(batch_size, group_size)
        flat_mask = mask.reshape(batch_size, group_size)
        latest_metrics = trainer.update(
            states,
            actions,
            old_log_probs,
            flat_rewards,
            flat_mask,
        )
        print(
            f"step={step + 1} "
            f"loss={latest_metrics['loss']:.4f} "
            f"ratio={latest_metrics['mean_ratio']:.4f} "
            f"kl={latest_metrics['mean_kl']:.4f}"
        )

    return latest_metrics


if __name__ == "__main__":
    metrics = run_demo()
    print("GRPO training finished!")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Mean ratio: {metrics['mean_ratio']:.4f}")
    print(f"Mean KL: {metrics['mean_kl']:.4f}")
