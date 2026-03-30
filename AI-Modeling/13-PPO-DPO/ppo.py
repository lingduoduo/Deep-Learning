# PPO Implementation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO:
    def __init__(
        self,
        policy,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        grad_clip=0.5,
    ):
        """
        policy: a network that includes:
                - Actor (outputs action probabilities)
                - Critic (outputs state value)
        clip_epsilon: PPO clipping range
        value_coef: weight for critic (value) loss
        entropy_coef: weight for entropy bonus (encourages exploration)
        grad_clip: maximum norm for gradient clipping
        """
        self.policy = policy
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.grad_clip = grad_clip

    @staticmethod
    def _flatten_last_dim(*tensors):
        return [tensor.squeeze(-1) for tensor in tensors]

    @staticmethod
    def _normalize(advantages):
        centered = advantages - advantages.mean()
        return centered / (advantages.std(unbiased=False) + 1e-8)

    def _compute_advantages(self, returns, values):
        advantages = returns - values.detach()
        return self._normalize(advantages)

    def _compute_actor_loss(self, log_probs, old_log_probs, advantages):
        ratios = torch.exp(log_probs - old_log_probs)
        clipped_ratios = torch.clamp(
            ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon
        )
        return -(torch.min(ratios * advantages, clipped_ratios * advantages)).mean()

    @staticmethod
    def _compute_critic_loss(values, returns):
        return F.mse_loss(values, returns)

    def compute_loss(self, states, actions, old_log_probs, returns):
        # 1. Get current policy log probabilities and value estimates
        log_probs = self.policy.get_log_prob(states, actions)
        values = self.policy.get_value(states)

        # Align common actor/critic output shapes such as [batch, 1] vs [batch].
        values, returns, old_log_probs, log_probs = self._flatten_last_dim(
            values, returns, old_log_probs, log_probs
        )

        # 2. Compute advantage and normalize it (important for stability)
        advantages = self._compute_advantages(returns, values)

        # 3. Actor loss (PPO clipped surrogate objective)
        actor_loss = self._compute_actor_loss(log_probs, old_log_probs, advantages)

        # 4. Critic loss (value function regression)
        critic_loss = self._compute_critic_loss(values, returns)

        # 5. Entropy bonus (encourages exploration)
        entropy = self.policy.get_entropy(states)

        # 6. Total loss
        total_loss = (
            actor_loss
            + self.value_coef * critic_loss
            - self.entropy_coef * entropy.mean()
        )

        return total_loss

    def update(self, states, actions, old_log_probs, returns):
        loss = self.compute_loss(states, actions, old_log_probs, returns)

        self.policy.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (prevents exploding gradients)
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.grad_clip
        )

        self.policy.optimizer.step()

        return loss.item()


class Policy(nn.Module):
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
        features = self.shared(states)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def _distribution(self, states):
        logits, _ = self.forward(states)
        return torch.distributions.Categorical(logits=logits)

    def get_log_prob(self, states, actions):
        return self._distribution(states).log_prob(actions)

    def get_value(self, states):
        _, value = self.forward(states)
        return value

    def get_entropy(self, states):
        return self._distribution(states).entropy()


def generate_dummy_data(batch_size=32, state_dim=4, action_dim=2):
    states = torch.randn(batch_size, state_dim)
    old_dist = torch.distributions.Categorical(
        logits=torch.randn(batch_size, action_dim)
    )
    actions = old_dist.sample()
    old_log_probs = old_dist.log_prob(actions)
    returns = torch.randn(batch_size)
    return states, actions, old_log_probs, returns


if __name__ == "__main__":
    state_dim = 4
    action_dim = 2

    policy = Policy(state_dim, action_dim)
    ppo = PPO(policy)

    states, actions, old_log_probs, returns = generate_dummy_data(
        state_dim=state_dim,
        action_dim=action_dim,
    )
    loss = ppo.update(states, actions, old_log_probs, returns)

    print("PPO training step done!")
    print(f"Loss: {loss:.4f}")
