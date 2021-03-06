"""Play Othello."""

import argparse
import othello
import simple_policies
from dqn import DQNAgent as DQN
from ppo import PPO
import numpy as np
import copy


def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1):
    if policy_type == 'rand':
        policy = simple_policies.RandomPolicy(seed=seed)
    elif policy_type == 'greedy':
        policy = simple_policies.GreedyPolicy()
    elif policy_type == 'maximin':
        policy = simple_policies.MaxiMinPolicy(search_depth)
    elif policy_type == 'human':
        policy = simple_policies.HumanPolicy(board_size)
    elif policy_type == 'dqn':
        policy = DQN('dqn', board_size)
    elif policy_type == 'ppo':
        policy = PPO('ppo', board_size)

    return policy


def make_state(obs, player_turn):
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

    state = np.stack([black, white, turn])
    return state

def play(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=100,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         num_disk_as_reward=False,
         render=True):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=protagonist_search_depth)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth)

    # disable .run
    def nop(*args):
        pass
    opponent_policy.run = nop
    if not hasattr(protagonist_policy, 'run'):
        protagonist_policy.run = nop

    # if opponent_agent_type == 'human':
    #     render_in_step = True
    # else:
    #     render_in_step = False

    env = othello.SimpleOthelloEnv(
            board_size=board_size,
            seed=rand_seed,
            initial_rand_steps=env_init_rand_steps,
            num_disk_as_reward=num_disk_as_reward,
            render_in_step=render)

    win_cnts = draw_cnts = lose_cnts = 0
    for i in range(num_rounds):
        switch = np.random.randint(2)
        if switch:
            protagonist = protagonist * -1

        policy = {}
        if protagonist == -1:
            pcolor = 'BLACK'
            policy['black'] = protagonist_policy
            policy['white'] = opponent_policy
        else:
            pcolor = 'WHITE'
            policy['black'] = opponent_policy
            policy['white'] = protagonist_policy

        print('Episode {}'.format(i + 1))
        print('Protagonist is {}'.format(pcolor))

        obs_b = env.reset()
        state_b = make_state(obs_b, env.player_turn)
        protagonist_policy.reset(env)
        opponent_policy.reset(env)
        if render:
            env.render()
        done_b = done_w = False
        init = True
        while not (done_b or done_w):
            # black
            assert env.player_turn == -1
            action_b = policy['black'].get_action(state_b)
            next_obs_b, reward_b, done_b, _ = env.step(action_b)
            next_state_b = make_state(next_obs_b, env.player_turn)
            while (not done_b) and env.player_turn == -1:
                policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                action_b = policy['black'].get_action(next_state_b)
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env.player_turn)

            # learning black policy
            if not init:
                policy['white'].run(state_w, action_w, - reward_b, done_b, next_state_b)
            init = False
            if done_b:
                policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                break

            # white
            assert env.player_turn == 1
            state_w = next_state_b
            action_w = policy['white'].get_action(state_w)
            next_obs_w, reward_w, done_w, _ = env.step(action_w)
            next_state_w = make_state(next_obs_w, env.player_turn)
            while (not done_w) and env.player_turn == 1:
                policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                action_w = policy['white'].get_action(next_state_w)
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env.player_turn)

            # learning black policy
            policy['black'].run(state_b, action_b, - reward_w, done_w, next_state_w)
            if done_w:
                policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                break

            state_b = next_state_w

            if render:
                env.render()

        if done_w:
            reward = reward_w * protagonist
        elif done_b:
            reward = reward_b * -protagonist
        else:
            raise ValueError

        print('reward={}'.format(reward))
        if num_disk_as_reward:
            total_disks = board_size ** 2
            if protagonist == 1:
                white_cnts = (total_disks + reward) / 2
                black_cnts = total_disks - white_cnts

                if white_cnts > black_cnts:
                    win_cnts += 1
                elif white_cnts == black_cnts:
                    draw_cnts += 1
                else:
                    lose_cnts += 1

            else:
                black_cnts = (total_disks + reward) / 2
                white_cnts = total_disks - black_cnts

                if black_cnts > white_cnts:
                    win_cnts += 1
                elif white_cnts == black_cnts:
                    draw_cnts += 1
                else:
                    lose_cnts += 1

        else:
            if reward == 1:
                win_cnts += 1
            elif reward == 0:
                draw_cnts += 1
            else:
                lose_cnts += 1
        print('-' * 3)
        print('#Wins: {}, #Draws: {}, #Loses: {}'.format(
            win_cnts, draw_cnts, lose_cnts))
    env.close()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo'])
    parser.add_argument('--protagonist-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--init-rand-steps', default=10, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    # Run test plays.
    protagonist = 1 if args.protagonist_plays_white else -1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    play(protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         protagonist_search_depth=args.protagonist_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         num_disk_as_reward=args.num_disk_as_reward,
         render=not args.no_render)

