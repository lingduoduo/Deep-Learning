import numpy as np
import sys

# Define action space
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Define terminal (treasure) location
DONE_LOCATION = 8

# Define GridWorld environment model
class GridWorldEnv():
    def __init__(self, shape=[5,5]):

        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape
        self.nS = np.prod(shape)     # number of states
        self.nA = 4                   # number of actions
        MAX_Y = shape[0]
        MAX_X = shape[1]

        P = {}
        grid = np.arange(self.nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(self.nA)}

            # s is an index; P[s][a] stores (prob, next_state, reward, done)
            # for each action UP/DOWN/LEFT/RIGHT

            is_done = lambda s: s == DONE_LOCATION

            # Reward function: all states = -1, terminal state = 0
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1

                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        self.isd = np.ones(self.nS)/self.nS   # initial state distribution
        self.P = P
        # super(GridWorldEnv,self).__init__(self.nS, self.nA, P, isd)

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Policy evaluation: compute the state-value function for a given policy.
    :param policy: the policy matrix (state × action probabilities)
    :param env: environment model containing transition probabilities and rewards
    :param discount_factor: discount factor γ
    :param theta: threshold for stopping iteration
    :return: state value function V
    """
    V = np.zeros(env.nS)

    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v

        if delta < theta:
            break

    return V


def get_max_index(action_values):
    """
    Given action-value estimates, return:
    - the indexes of actions with maximal value
    - a policy vector marking the best actions with probability 1
    """
    indexes = []
    policy_arr = np.zeros(len(action_values))
    max_action_value = np.max(action_values)

    for i in range(len(action_values)):
        action_value = action_values[i]
        if action_value == max_action_value:
            indexes.append(i)
            policy_arr[i] = 1.0

    return indexes, policy_arr


def change_policy(policys):
    action_tuple = []
    for policy in policys:
        action_tuple.append(tuple(get_max_index(policy)))
    return action_tuple


def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy improvement using policy iteration:
    1. Start with a random policy
    2. Evaluate the policy
    3. Improve it by acting greedily w.r.t the value function
    4. Repeat until the policy stabilizes
    """
    policy = np.ones([env.nS, env.nA]) / env.nA   # initialize with random policy

    while True:
        V = policy_eval_fn(policy, env, discount_factor)
        policy_stable = True

        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])  # current action under policy

            # Compute action-values Q(s,a)
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])

            best_a, best_policy = get_max_index(action_values)

            # If current action is not optimal, policy is unstable
            if chosen_a not in best_a:
                policy_stable = False

            policy[s] = best_policy

        if policy_stable:
            break

    return policy


if __name__ == '__main__':
    env = GridWorldEnv()
    print(env.P)
    # policy = policy_improvement(env)
    # print(policy)


