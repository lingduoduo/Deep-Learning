import copy

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. Fundamental GRPO Utility Functions
# ============================================================

def masked_mean(values, mask):
    """
    Compute the mean over valid positions only.

    values: Tensor, shape (batch_size, group_size)
    mask: Tensor, shape (batch_size, group_size)
    """
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (values * mask).sum(dim=1, keepdim=True) / denom


def compute_group_advantages(rewards, mask):
    """
    GRPO normalizes rewards within each group instead of relying on a critic.

    rewards: Tensor, shape (batch_size, group_size)
    mask: Tensor, shape (batch_size, group_size)

    returns:
        advantages: Tensor, shape (batch_size, group_size)
    """
    mask = mask.float()

    group_mean = masked_mean(rewards, mask)
    centered_rewards = (rewards - group_mean) * mask

    group_variance = masked_mean(centered_rewards.pow(2), mask)
    group_std = torch.sqrt(group_variance + 1e-8)

    advantages = centered_rewards / group_std

    return advantages.detach()


def grpo_clipped_policy_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_epsilon=0.2,
):
    """
    PPO-style clipped objective used by GRPO.
    """
    ratios = torch.exp(new_log_probs - old_log_probs)

    unclipped_objective = ratios * advantages

    clipped_ratios = torch.clamp(
        ratios,
        1 - clip_epsilon,
        1 + clip_epsilon,
    )
    clipped_objective = clipped_ratios * advantages

    policy_loss = -torch.min(
        unclipped_objective,
        clipped_objective,
    )

    return policy_loss, ratios


def reverse_kl_penalty(policy_log_probs, ref_log_probs):
    """
    Sample-based reverse-KL style penalty often used in RLHF variants.
    """
    log_ratio = policy_log_probs - ref_log_probs
    penalty = torch.exp(log_ratio) - log_ratio - 1.0
    return penalty


# ============================================================
# 2. Policy Network
# ============================================================

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

    def get_distribution(self, states):
        logits = self.forward(states)
        return torch.distributions.Categorical(logits=logits)

    def get_log_probs(self, states, actions):
        dist = self.get_distribution(states)
        return dist.log_prob(actions)


# ============================================================
# 3. GRPO Trainer
# ============================================================

class GRPOTrainer:
    def __init__(
        self,
        policy,
        reference_policy,
        clip_epsilon=0.2,
        beta=0.1,
        grad_clip=1.0,
    ):
        self.policy = policy
        self.reference_policy = reference_policy
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.grad_clip = grad_clip

        self.reference_policy.eval()
        for param in self.reference_policy.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        states,
        actions,
        old_log_probs,
        rewards,
        mask,
    ):
        """
        states: Tensor, shape (batch_size * group_size, state_dim)
        actions: Tensor, shape (batch_size * group_size,)
        old_log_probs: Tensor, shape (batch_size * group_size,)
        rewards: Tensor, shape (batch_size, group_size)
        mask: Tensor, shape (batch_size, group_size)
        """
        flat_mask = mask.reshape(-1).float()
        advantages = compute_group_advantages(rewards, mask).reshape(-1)

        new_log_probs = self.policy.get_log_probs(states, actions)

        with torch.no_grad():
            ref_log_probs = self.reference_policy.get_log_probs(states, actions)

        policy_loss, ratios = grpo_clipped_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )

        kl_penalty = reverse_kl_penalty(
            policy_log_probs=new_log_probs,
            ref_log_probs=ref_log_probs,
        )

        total_loss = policy_loss + self.beta * kl_penalty
        mean_loss = (total_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1e-8)

        metrics = {
            "loss": mean_loss.item(),
            "mean_reward": (
                (rewards.reshape(-1) * flat_mask).sum()
                / flat_mask.sum().clamp_min(1e-8)
            ).item(),
            "mean_advantage": (
                (advantages * flat_mask).sum()
                / flat_mask.sum().clamp_min(1e-8)
            ).item(),
            "mean_ratio": (
                (ratios * flat_mask).sum()
                / flat_mask.sum().clamp_min(1e-8)
            ).item(),
            "mean_kl": (
                (kl_penalty * flat_mask).sum()
                / flat_mask.sum().clamp_min(1e-8)
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
    ):
        loss, metrics = self.compute_loss(
            states,
            actions,
            old_log_probs,
            rewards,
            mask,
        )

        self.policy.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.grad_clip,
        )

        self.policy.optimizer.step()

        return metrics


# ============================================================
# 4. Dummy Grouped Bandit Example
# ============================================================

def reward_function(states, actions, reward_matrix):
    """
    Synthetic reward:
    each state has a hidden preferred action determined by reward_matrix.
    """
    target_logits = states @ reward_matrix
    target_actions = target_logits.argmax(dim=-1)

    rewards = (actions == target_actions).float()

    # Small shaping term makes the signal less binary.
    rewards = rewards + 0.05 * target_logits.gather(
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)

    return rewards


def generate_dummy_group_data(
    old_policy,
    reward_matrix,
    batch_size=16,
    group_size=6,
    state_dim=8,
):
    """
    Build grouped samples where each prompt/state is repeated group_size times.
    """
    base_states = torch.randn(batch_size, state_dim)

    grouped_states = base_states.unsqueeze(1).repeat(1, group_size, 1)
    flat_states = grouped_states.reshape(batch_size * group_size, state_dim)

    with torch.no_grad():
        old_dist = old_policy.get_distribution(flat_states)
        actions = old_dist.sample()
        old_log_probs = old_dist.log_prob(actions)

    rewards = reward_function(flat_states, actions, reward_matrix)
    rewards = rewards.reshape(batch_size, group_size)

    mask = torch.ones(batch_size, group_size)

    return flat_states, actions, old_log_probs, rewards, mask


# ============================================================
# 5. Training Demo
# ============================================================

def run_demo(
    steps=40,
    batch_size=32,
    group_size=8,
    state_dim=8,
    action_dim=4,
    hidden_dim=64,
    lr=1e-2,
):
    torch.manual_seed(42)

    policy = Policy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=lr,
    )

    reference_policy = copy.deepcopy(policy)

    trainer = GRPOTrainer(
        policy=policy,
        reference_policy=reference_policy,
        clip_epsilon=0.2,
        beta=0.05,
        grad_clip=1.0,
    )

    reward_matrix = torch.randn(state_dim, action_dim)

    latest_metrics = None

    for step in range(steps):
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad = False

        states, actions, old_log_probs, rewards, mask = generate_dummy_group_data(
            old_policy=old_policy,
            reward_matrix=reward_matrix,
            batch_size=batch_size,
            group_size=group_size,
            state_dim=state_dim,
        )

        latest_metrics = trainer.update(
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            rewards=rewards,
            mask=mask,
        )

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"step={step + 1:02d} "
                f"loss={latest_metrics['loss']:.4f} "
                f"reward={latest_metrics['mean_reward']:.4f} "
                f"ratio={latest_metrics['mean_ratio']:.4f} "
                f"kl={latest_metrics['mean_kl']:.4f}"
            )

    return latest_metrics


if __name__ == "__main__":
    metrics = run_demo()

    print("\nGRPO training finished!")
    print(f"Final loss: {metrics['loss']:.4f}")
    print(f"Mean reward: {metrics['mean_reward']:.4f}")
    print(f"Mean ratio: {metrics['mean_ratio']:.4f}")
    print(f"Mean KL: {metrics['mean_kl']:.4f}")
