import numpy as np
import copy
import torch
from dqn import DQNAgent as DQN
from ppo import PPO
from Rainbow.agent import Agent as Rainbow


def save(episode, policy, loss, path):
    # torch.save({
    #     'episode': episode,
    #     'model_state_dict': policy.policy_old.state_dict(),
    #     'optimizer_state_dict': policy.optimizer.state_dict(),
    #     'loss': loss,
    # }, path)
    policy.save(path, episode, loss)


def load(policy, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    policy.load(checkpoint)
    episode = checkpoint['episode']
    loss = checkpoint['loss']
    return episode, loss


def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1, agent_name=''):
    import simple_policies
    if policy_type == 'rand':
        policy = simple_policies.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = simple_policies.GreedyPolicy()
    elif policy_type == 'maximin':
        policy = simple_policies.MaxiMinPolicy(search_depth)
    elif policy_type == 'human':
        policy = simple_policies.HumanPolicy(board_size)
    elif policy_type == 'dqn':
        policy = DQN(agent_name, board_size)
    elif policy_type == 'ppo':
        policy = PPO(agent_name, board_size)
    elif policy_type == 'rainbow':
        policy = Rainbow(agent_name, board_size)

    return policy


def make_state(obs, env):
    player_turn = env.player_turn
    moves_number = np.array(env.possible_moves)
    size = len(obs)
    idx1 = moves_number // size
    idx2 = moves_number % size
    possible_moves = np.zeros(obs.shape)
    if len(idx1) > 0 and len(idx2) > 1:
        possible_moves[idx1, idx2] = 1

    black = copy.deepcopy(obs)
    white = copy.deepcopy(obs)
    # black
    if player_turn == -1:
        turn = np.zeros(obs.shape)
        black[black == -1] = 0
        white[white == 1] = 0
        white[white == -1] = 1
    # white
    else:
        turn = np.ones(obs.shape)
        black[black == 1] = 0
        black[black == -1] = 1
        white[white == -1] = 0

    state = np.stack([black, white, turn, possible_moves])
    return state


def undo_state(state, player_turn):
    assert int((player_turn+1) / 2) == int(state[2][0][0])
    # black
    if player_turn == -1:
        obs = state[0] - state[1]
    # white
    else:
        obs = state[1] - state[0]
    return obs
