"""Play Othello."""

import argparse
import othello
import simple_policies
from dqn import DQNAgent as DQN
from ppo import PPO
import numpy as np
import copy
import queue
from util import make_state, create_policy, save, load


def action(color, p_color, state, policy):
    if color == p_color:
        action = policy[color].get_action(state)
    else:
        action = policy[color].get_test_action(state)
    return action


def play(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=100,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         test_init_rand_steps=10,
         num_disk_as_reward=True,
         render=False,
         test_interval=2500,
         num_test_games=200,
         save_interval=5000,
         # load_path='data/selfplay/rainbow_selfplay_350000.pth'):
         load_path=''):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    agent_name = 'rainbow_selfplay_2nd'

    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=protagonist_search_depth,
        agent_name=agent_name)
    opponent_policy1 = create_policy(
        policy_type='rand',
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth)
    opponent_policy2 = create_policy(
        policy_type='greedy',
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth)
    opponent_policies = [('rand', opponent_policy1), ('greedy', opponent_policy2)]

    # disable .run
    # def nop(*args):
    #     pass
    # opponent_policy.run = nop
    # if not hasattr(protagonist_policy, 'run'):
    #     protagonist_policy.run = nop

    # if opponent_agent_type == 'human':
    #     render_in_step = True
    # else:
    #     render_in_step = False

    if load_path:
        print('Load {} ...'.format(load_path))
        start_episode, loss = load(protagonist_policy, load_path)
    else:
        start_episode = 0


    env = othello.SimpleOthelloEnv(
            board_size=board_size,
            seed=rand_seed,
            initial_rand_steps=env_init_rand_steps,
            num_disk_as_reward=num_disk_as_reward,
            render_in_step=render)

    win_cnts = draw_cnts = lose_cnts = 0
    for i in range(start_episode, num_rounds):
        switch = np.random.randint(2)
        if switch:
            protagonist = protagonist * -1

        policy = {}
        if protagonist == -1:
            pcolor = 'black'
            policy['black'] = protagonist_policy
            policy['white'] = protagonist_policy
        else:
            pcolor = 'white'
            policy['black'] = protagonist_policy
            policy['white'] = protagonist_policy

        print('Episode {}'.format(i + 1))
        print('Protagonist is {}'.format(pcolor))

        obs_b = env.reset()
        state_b = make_state(obs_b, env)
        protagonist_policy.reset(env)
        # opponent_policy.reset(env)
        if render:
            env.render()
        done_b = done_w = False
        init = True
        while not (done_b or done_w):
            # black
            assert env.player_turn == -1
            # action_b = policy['black'].get_action(state_b)
            action_b = action('black', pcolor, state_b, policy)
            next_obs_b, reward_b, done_b, _ = env.step(action_b)
            next_state_b = make_state(next_obs_b, env)
            while (not done_b) and env.player_turn == -1:
                if pcolor == 'black':
                    policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                # action_b = policy['black'].get_action(next_state_b)
                action_b = action('black', pcolor, next_state_b, policy)
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)

            # learning black policy
            if not init:
                if pcolor == 'white':
                    policy['white'].run(state_w, action_w, - reward_b, done_b, next_state_b)
            init = False
            if done_b:
                if pcolor == 'black':
                    policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                break

            # white
            assert env.player_turn == 1
            state_w = next_state_b
            # action_w = policy['white'].get_action(state_w)
            action_w = action('white', pcolor, state_w, policy)
            next_obs_w, reward_w, done_w, _ = env.step(action_w)
            next_state_w = make_state(next_obs_w, env)
            while (not done_w) and env.player_turn == 1:
                if pcolor == 'white':
                    policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                # action_w = policy['white'].get_action(next_state_w)
                action_w = action('white', pcolor, next_state_w, policy)
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env)

            # learning black policy
            if pcolor == 'black':
                policy['black'].run(state_b, action_b, - reward_w, done_w, next_state_w)
            if done_w:
                if pcolor == 'white':
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

        # calc student's winning %
        if i % test_interval == 0:
            env.initial_rand_steps = test_init_rand_steps
            for name, opponent_policy in opponent_policies:
                wins = 0
                protagonist = -1
                for j in range(num_test_games):
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

                    obs_b = env.reset()
                    state_b = make_state(obs_b, env)
                    protagonist_policy.reset(env)
                    opponent_policy.reset(env)
                    if render:
                        env.render()
                    done_b = done_w = False
                    while not (done_b or done_w):
                        # black
                        assert env.player_turn == -1
                        action_b = policy['black'].get_test_action(state_b)
                        next_obs_b, reward_b, done_b, _ = env.step(action_b)
                        next_state_b = make_state(next_obs_b, env)
                        while (not done_b) and env.player_turn == -1:
                            # policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                            action_b = policy['black'].get_test_action(next_state_b)
                            next_obs_b, reward_b, done_b, _ = env.step(action_b)
                            next_state_b = make_state(next_obs_b, env)
                        if done_b:
                            break

                        # white
                        assert env.player_turn == 1
                        state_w = next_state_b
                        action_w = policy['white'].get_test_action(state_w)
                        next_obs_w, reward_w, done_w, _ = env.step(action_w)
                        next_state_w = make_state(next_obs_w, env)
                        while (not done_w) and env.player_turn == 1:
                            # policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                            action_w = policy['white'].get_test_action(next_state_w)
                            next_obs_w, reward_w, done_w, _ = env.step(action_w)
                            next_state_w = make_state(next_obs_w, env)
                        if done_w:
                            break
                        state_b = next_state_w

                    if done_w:
                        reward = reward_w * protagonist
                    elif done_b:
                        reward = reward_b * -protagonist
                    else:
                        raise ValueError
                    if reward > 0:
                        wins += 1
                # last_win_per = win_per
                win_per = wins / num_test_games
                print()
                print('win % ({}):'.format(name), win_per)
                print()
                protagonist_policy.writer.add_scalar("win%({})".format(name), win_per, i)
                env.initial_rand_steps = env_init_rand_steps

        if i % save_interval == 0:
            save_path = '/data/unagi0/omura/othello/selfplay/{}_{}.pth'.format(agent_name, i)
            # save_path = 'data/selfplay/{}_{}.pth'.format(agent_name, i)
            save(i, protagonist_policy, 0, save_path)
    env.close()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo', 'rainbow'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo', 'rainbow'])
    parser.add_argument('--protagonist-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
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

