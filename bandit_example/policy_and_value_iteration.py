# Spring 2023, 535515 Reinforcement Learning
# HW0: Policy Iteration and Value iteration for MDPs
       
import numpy as np
from bandit import Bandit

FEEDBACK_DIM = 20
STATE_DIM = 21
ACTION_DIM = 21
N = int((ACTION_DIM - 1) / 2)

def get_rewards_and_transitions_from_env(env):
    # Intiailize matrices
    R = np.zeros((STATE_DIM, ACTION_DIM))
    P = np.zeros((STATE_DIM, ACTION_DIM, STATE_DIM))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s1 in range(-N, N+1):
        for a in range(-N, N+1):
            for s2 in range(-N, N+1):
                s1_index = action_state_to_index(s1)
                a_index = action_state_to_index(a)
                s2_index = action_state_to_index(s2)
                R[s1_index, a_index] = -np.power(s1 - a, 2)
                P[s1_index, a_index, s2_index] = 1 / ACTION_DIM
    
    return R, P

def action_state_to_index(state_action):
    return state_action + N

def calculate_policy_transition_and_reward(P, R, policy, num_actions):
    policy_2d = np.eye(num_actions)[policy]
    return np.sum(policy_2d @ P, 1), np.sum(np.multiply(policy_2d, R), 1)

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    number_range = np.array([num for num in range(-N, N+1)])
    policy = np.array([np.random.choice(number_range) for _ in range(-N, N+1)])
    
    value_function = np.random.randn(STATE_DIM) * 10
    R, P = get_rewards_and_transitions_from_env(env)

    k = 0          
    while k < max_iterations:        
        current_value_function = np.copy(value_function)
        # Evalutate and improve value function
        for s_index in range(STATE_DIM):
            current_rewards = R[s_index] + (gamma * (P[s_index] @ current_value_function))
            value_function[s_index] = np.max(current_rewards)

        # Check if change in value function is non-trivial
        if np.linalg.norm(current_value_function - value_function, ord=2) < eps:
            print(k, np.linalg.norm(current_value_function - value_function, ord=2))
            break

        k += 1
        if k % 100 == 0:
            print("Value Iteration:", k, np.linalg.norm(current_value_function - value_function, ord=2))
    
    # Derive optimal policy
    for s_index in range(STATE_DIM):
        q_values = R[s_index] + (gamma * (P[s_index] @ value_function))
        policy[s_index] = np.argmax(q_values)
    
    return policy - N

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    number_range = np.array([num for num in range(-N, N+1)])
    policy = np.array([np.random.choice(number_range) for _ in range(-N, N+1)])
    value_function = np.random.randn(STATE_DIM) * 10
    
    R, P = get_rewards_and_transitions_from_env(env)

    j = 0
    while j < 1000:                
        k = 0          
        while k < max_iterations:        
            # Evalutate and improve value function
            current_value_function = np.copy(value_function)
            for s_index in range(STATE_DIM):
                current_reward = R[s_index] + (gamma * (P[s_index] @ current_value_function))
                value_function[s_index] = current_reward[policy[s_index]]

            if np.linalg.norm(current_value_function - value_function, ord=2) < eps:
                break
            k += 1
        
        # Policy Iteration
        prev_policy = np.copy(policy)
        for s_index in range(STATE_DIM):
            q_values = R[s_index] + (gamma * (P[s_index] @ value_function))
            policy[s_index] = np.argmax(q_values)
        
        if np.linalg.norm(policy - prev_policy, ord=2) < eps:
            print(j, np.linalg.norm(policy - prev_policy, ord=2))
            break
        
        j += 1

    return policy - N

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = Bandit(K=FEEDBACK_DIM, N=N, variance=10)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy

if __name__ == '__main__':
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v3')

    print(pi_policy)
    print(vi_policy)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)