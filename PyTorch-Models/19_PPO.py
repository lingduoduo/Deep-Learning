import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 1. Fundamental PPO Utility Functions
# ============================================================

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation.

    rewards: Tensor, shape (T,)
    values: Tensor, shape (T + 1,)
            Last value is V(s_{T+1})

    returns:
        advantages: Tensor, shape (T,)
        returns: Tensor, shape (T,)
    """
    T = len(rewards)
    advantages = torch.zeros(T)

    gae = 0.0

    for t in reversed(range(T)):
        td_error = rewards[t] + gamma * values[t + 1] - values[t]
        gae = td_error + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values[:-1]

    return advantages, returns


def normalize_advantages(advantages):
    """
    Normalize advantages for more stable PPO training.
    """
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def ppo_clipped_policy_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    clip_epsilon=0.2,
):
    """
    PPO clipped surrogate objective.

    ratio = pi_new(a|s) / pi_old(a|s)

    In log space:
    ratio = exp(log_pi_new - log_pi_old)
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
    ).mean()

    return policy_loss


def value_loss(values, returns):
    """
    Critic loss: regress predicted values toward target returns.
    """
    return F.mse_loss(values, returns)


def entropy_bonus(entropy):
    """
    Entropy encourages exploration.
    """
    return entropy.mean()

# ============================================================
# 2. Actor-Critic Policy Network
# ============================================================

class ActorCriticPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=1e-3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, states):
        """
        states: shape (batch_size, state_dim)

        returns:
            logits: shape (batch_size, action_dim)
            values: shape (batch_size,)
        """
        features = self.shared(states)

        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)

        return logits, values

    def get_distribution(self, states):
        logits, _ = self.forward(states)
        return torch.distributions.Categorical(logits=logits)

    def get_log_probs(self, states, actions):
        dist = self.get_distribution(states)
        return dist.log_prob(actions)

    def get_entropy(self, states):
        dist = self.get_distribution(states)
        return dist.entropy()

    def get_values(self, states):
        _, values = self.forward(states)
        return values

# ============================================================
# 3. PPO Trainer
# ============================================================

class PPOTrainer:
    def __init__(
        self,
        policy,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_clip=0.5,
    ):
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

    def compute_loss(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
    ):
        """
        Compute PPO total loss.

        states: shape (batch_size, state_dim)
        actions: shape (batch_size,)
        old_log_probs: shape (batch_size,)
        returns: shape (batch_size,)
        advantages: shape (batch_size,)
        """

        # 1. Current policy log probabilities
        new_log_probs = self.policy.get_log_probs(states, actions)

        # 2. Current value predictions
        values = self.policy.get_values(states)

        # 3. Entropy
        entropy = self.policy.get_entropy(states)

        # 4. Normalize advantages
        advantages = normalize_advantages(advantages)

        # 5. Actor loss
        actor_loss = ppo_clipped_policy_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
        )

        # 6. Critic loss
        critic_loss = value_loss(values, returns)

        # 7. Entropy bonus
        entropy = entropy_bonus(entropy)

        # 8. Final PPO loss
        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy
        )

        return total_loss, actor_loss, critic_loss, entropy

    def update(
        self,
        states,
        actions,
        old_log_probs,
        returns,
        advantages,
    ):
        total_loss, actor_loss, critic_loss, entropy = self.compute_loss(
            states,
            actions,
            old_log_probs,
            returns,
            advantages,
        )

        self.policy.optimizer.zero_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(),
            self.grad_clip,
        )

        self.policy.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item(),
        }

# ============================================================
# 4. Dummy Data Example
# ============================================================

def generate_dummy_data(batch_size=32, state_dim=4, action_dim=2):
    states = torch.randn(batch_size, state_dim)

    old_logits = torch.randn(batch_size, action_dim)
    old_dist = torch.distributions.Categorical(logits=old_logits)

    actions = old_dist.sample()
    old_log_probs = old_dist.log_prob(actions)

    returns = torch.randn(batch_size)
    advantages = torch.randn(batch_size)

    return states, actions, old_log_probs, returns, advantages


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2

    policy = ActorCriticPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=64,
        lr=1e-3,
    )

    trainer = PPOTrainer(policy)

    states, actions, old_log_probs, returns, advantages = generate_dummy_data(
        state_dim=state_dim,
        action_dim=action_dim,
    )

    logs = trainer.update(
        states=states,
        actions=actions,
        old_log_probs=old_log_probs,
        returns=returns,
        advantages=advantages,
    )

    print("PPO update completed!")
    print(logs)