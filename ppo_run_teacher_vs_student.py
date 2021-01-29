"""Play Othello."""

import os
import argparse
import othello
import simple_policies
from dqn import DQNAgent as DQN
from ppo import PPO
import numpy as np
import copy
import queue
from util import make_state, create_policy, save, load

from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr import algo, utils
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.algo import gail
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.arguments import get_args
# from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.envs import make_vec_envs
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.model import Policy
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.storage import RolloutStorage

import torch

import multiprocessing as mp
import random
from gym import spaces
from collections import deque

from torch.utils.tensorboard import SummaryWriter

from envs import PPOEnvs

from multiprocessing import set_start_method
#
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass



def test(protagonist,
         protagonist_agent_type='greedy',
         opponent_agent_type='rand',
         board_size=8,
         num_rounds=200000,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         test_init_rand_steps=10,
         num_disk_as_reward=True,
         render=False,
         test_interval=500,
         num_test_games=200,
         save_interval=500,
         # load_path='data/selfplay/rainbow_selfplay_350000.pth'):
         load_path=''):

    args = get_args()
    args.algo = 'ppo'
    args.use_gae = True
    args.lr = 5e-5 #2.5e-4
    args.clip_param = 0.2
    args.value_loss_coef = 0.5 #0.5
    args.num_processes = 8
    args.num_steps = 8 #128
    args.num_mini_batch = 4
    args.log_interval = 1
    args.use_linear_lr_decay = True
    args.entropy_coef = 0 # 0.01
    print(args)

    step_per_episode = 32
    # num_rounds_per_proc = num_rounds // args.num_processes
    num_updates = (num_rounds * step_per_episode) // args.num_steps

    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    #
    # if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #
    # log_dir = os.path.expanduser(args.log_dir)
    # eval_log_dir = log_dir + "_eval"
    # utils.cleanup_log_dir(log_dir)
    # utils.cleanup_log_dir(eval_log_dir)

    # torch.set_num_threads(1)
    # device = torch.device("cuda:0" if args.cuda else "cpu")
    device = torch.device("cpu")

    # agent_name = 'ppo_selfplay_8proc_th1e-10_ent1e-2'
    agent_name = 'ent0_lr1e-5_clip2e-1'
    writer = SummaryWriter(log_dir="./log/ppo_selfplay/{}".format(agent_name))

    envs_list = []
    for i in range(args.num_processes):
        env = othello.SimpleOthelloEnv(
            board_size=board_size,
            seed=i,
            initial_rand_steps=env_init_rand_steps,
            num_disk_as_reward=num_disk_as_reward,
            render_in_step=render)
        env.rand_steps_holder = env_init_rand_steps
        env.test_rand_steps_holder = test_init_rand_steps
        envs_list.append(env)

    obs_space = spaces.Box(np.zeros((4, 8, 8)), np.ones((4, 8, 8)))
    action_space = spaces.Discrete(board_size ** 2)

    if load_path:
        actor_critic = torch.load(load_path)
    else:
        actor_critic = Policy(
            obs_space.shape,
            action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)

    envs = PPOEnvs(envs_list, subproc_worker, actor_critic, device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    # episode_rewards = deque(maxlen=10)
    u = 0
    for episode in range(num_rounds):
        print()
        print('Episode %s' % episode)
        envs.reset()
        over = False
        while not over:
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, u, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)
            u += 1
            # print(rollouts.rewards.squeeze().squeeze())
            for step in range(args.num_steps):

                # Obser reward and next obs
                if not over:
                    obs, action, reward, done, infos, v_logprob_hidden, masks, bad_masks = envs.step(rollouts.recurrent_hidden_states[step])
                    choices = [info['choices'] for info in infos]
                    for i in range(len(action)):
                        assert done[i] or action[i][0] in choices[i], (action[i][0], choices[i])
                # for info in infos:
                #     if 'episode' in info.keys():
                #         episode_rewards.append(info['episode']['r'])

                if step == 0:
                    rollouts.obs[0].copy_(obs)
                    rollouts.masks[0].copy_(masks)
                    rollouts.bad_masks[0].copy_(bad_masks)
                else:
                    rollouts.insert(obs, prev_hidden, prev_action,
                                    prev_logprob, prev_value, prev_reward, masks, bad_masks, prev_choices)
                # prev_obs = obs
                prev_action = action
                prev_value = v_logprob_hidden[:, 0].unsqueeze(1)
                prev_logprob = v_logprob_hidden[:, 1].unsqueeze(1)
                prev_hidden = v_logprob_hidden[:, 2].unsqueeze(1)
                prev_reward = reward
                # prev_masks = masks
                # prev_bad_masks = bad_masks
                prev_choices = choices
                over = all(done)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            rollouts.after_update()

        if episode % test_interval == 0:
            games, wins = envs.test('rand', num_test_games, rollouts.recurrent_hidden_states[0])
            writer.add_scalar("win%({})".format('rand'), wins/games, episode)
            print('### vs-random winning% {}/{}={}'.format(wins, games, wins / games))
            games, wins = envs.test('greedy', num_test_games, rollouts.recurrent_hidden_states[0])
            writer.add_scalar("win%({})".format('greedy'), wins/games, episode)
            print('### vs-greedy winning% {}/{}={}'.format(wins, games, wins/games))
        if episode % save_interval == 0:
            if os.path.exists('/data/unagi0/omura'):
                save_path = '/data/unagi0/omura/othello/selfplay/{}_{}.pth'.format(agent_name, episode)
            else:
                save_path = 'data/selfplay/{}_{}.pth'.format(agent_name, episode)
            torch.save(actor_critic, save_path)

        writer.add_scalar("value_loss", value_loss, episode)
        writer.add_scalar("action_loss", action_loss, episode)
        writer.add_scalar("dist_entropy", dist_entropy, episode)
        print(value_loss, action_loss, dist_entropy)

    envs.over()


def random_possible_actions(infos):
    actions = []
    for info in infos:
        if len(info['choices']) != 0:
            actions.append(random.choice(info['choices']))
        else:
            actions.append(-1)
    return actions


def subproc_worker(id, env, pipe, parent_pipe):
    parent_pipe.close()
    o = np.zeros((4, env.board_size, env.board_size))
    dummy_outputs = (0, 0, 0)

    done = 0

    i = 0
    recv = True
    while True:
        i += 1
        if recv:
            cmd, a = pipe.recv()
        else:
            recv = True

        if cmd == 'over':
            break
        elif cmd == 'reset':
            obs_b = env.reset()
            state_b = make_state(obs_b, env)
            protagonist = np.random.randint(2)
            protagonist = -1 if protagonist == 0 else 1
            pcolor = 'black' if protagonist == -1 else 'white'
            done = False
            done_b = done_w = False
            init = True

        elif cmd == 'step':
            if done:
                pipe.send((o, 0, 0, done, {'type': 'over', 'choices': env.possible_moves}, dummy_outputs))
                cmd, a = pipe.recv()
                if cmd == 'reset':
                    recv = False
                elif cmd == 'over':
                    break
                continue

            while not (done_b or done_w):
                # black
                assert env.player_turn == -1
                pipe.send((state_b, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_b, output_b = pipe.recv()
                choice_b = env.possible_moves
                assert cmd == 'step'
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)
                while (not done_b) and env.player_turn == -1:
                    if pcolor == 'black':
                        pipe.send((state_b, action_b, reward_b, done_b, {'type': None, 'choices': choice_b}, output_b))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    pipe.send((next_state_b, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_b, output_b = pipe.recv()
                    choice_b = env.possible_moves
                    assert cmd == 'step'
                    next_obs_b, reward_b, done_b, _ = env.step(action_b)
                    next_state_b = make_state(next_obs_b, env)

                # learning black policy
                if not init:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, - reward_b, done_b, {'type': None, 'choices': choice_w}, output_w))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'

                if done_b:
                    if pcolor == 'black':
                        pipe.send((state_b, action_b, reward_b, done_b, {'type': None, 'choices': env.possible_moves}, output_b))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    if pcolor == 'white' and init:
                        pipe.send((o, 0, 0, done_b, {'type': None, 'choices': env.possible_moves}, dummy_outputs))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    break
                init = False

                # white
                assert env.player_turn == 1
                state_w = next_state_b
                pipe.send((state_w, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_w, output_w = pipe.recv()
                choice_w = env.possible_moves
                assert cmd == 'step'
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env)
                while (not done_w) and env.player_turn == 1:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, reward_w, done_w, {'type': None, 'choices': choice_w}, output_w))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    pipe.send((next_state_w, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_w, output_w = pipe.recv()
                    choice_w = env.possible_moves
                    assert cmd == 'step'
                    next_obs_w, reward_w, done_w, _ = env.step(action_w)
                    next_state_w = make_state(next_obs_w, env)

                # learning black policy
                if pcolor == 'black':
                    pipe.send((state_b, action_b, - reward_w, done_w, {'type': None, 'choices': choice_b}, output_b))
                    cmd, pipe.recv()[0]
                    assert cmd == 'step'
                if done_w:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, reward_w, done_w, {'type': None, 'choices': choice_w}, output_w))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    break
                state_b = next_state_w
            done = True

        elif cmd == 'test-rand':
            num_wins = rule_base_game('rand', a, env, pipe)
            while cmd != 'finish-test':
                pipe.send(
                    (o, 0, 0, 0, {'type': 'over', 'choices': env.possible_moves, 'wins': num_wins}, dummy_outputs))
                cmd, _, _ = pipe.recv()
        elif cmd == 'test-greedy':
            num_wins = rule_base_game('greedy', a, env, pipe)
            while cmd != 'finish-test':
                pipe.send(
                    (o, 0, 0, 0, {'type': 'over', 'choices': env.possible_moves, 'wins': num_wins}, dummy_outputs))
                cmd, _, _ = pipe.recv()


def rule_base_game(name, num_games, env, pipe):
    env.initial_rand_steps = env.test_rand_steps_holder
    dummy_outputs = (0, 0, 0)

    policy = create_policy(
        policy_type=name,
        board_size=env.board_size,
        seed=env.rand_seed)

    def get_action(color, p_color, state):
        if color == p_color:
            pipe.send((state, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
            _, action, _ = pipe.recv()
        else:
            action = policy.get_test_action(state)
        return action

    num_wins = 0
    for _ in range(num_games):
        # reset
        obs_b = env.reset()
        policy.reset(env)
        state_b = make_state(obs_b, env)
        protagonist = np.random.randint(2)
        protagonist = -1 if protagonist == 0 else 1
        pcolor = 'black' if protagonist == -1 else 'white'
        done = False
        done_b = done_w = False
        init = True

        # game
        while not (done_b or done_w):
            assert env.player_turn == -1
            action_b = get_action(env.player_turn, protagonist, state_b)
            next_obs_b, reward_b, done_b, _ = env.step(action_b)
            next_state_b = make_state(next_obs_b, env)
            while (not done_b) and env.player_turn == -1:
                action_b = get_action(env.player_turn, protagonist, next_state_b)

                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)

            if done_b:
                break
            init = False

            # white
            assert env.player_turn == 1
            state_w = next_state_b
            action_w = get_action(env.player_turn, protagonist, state_w)
            next_obs_w, reward_w, done_w, _ = env.step(action_w)
            next_state_w = make_state(next_obs_w, env)
            while (not done_w) and env.player_turn == 1:
                action_w = get_action(env.player_turn, protagonist, next_state_w)
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
            num_wins += 1
    env.initial_rand_steps = env.rand_steps_holder
    return num_wins


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
    parser.add_argument('--num-rounds', default=50000, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
    parser.add_argument('--render', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    # Run test plays.
    protagonist = 1 if args.protagonist_plays_white else -1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    # play(protagonist=protagonist,
    #      protagonist_agent_type=protagonist_agent_type,
    #      opponent_agent_type=opponent_agent_type,
    #      board_size=args.board_size,
    #      num_rounds=args.num_rounds,
    #      protagonist_search_depth=args.protagonist_search_depth,
    #      opponent_search_depth=args.opponent_search_depth,
    #      rand_seed=args.rand_seed,
    #      env_init_rand_steps=args.init_rand_steps,
    #      num_disk_as_reward=args.num_disk_as_reward,
    #      render=not args.no_render)
    # test()
    test(protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         protagonist_search_depth=args.protagonist_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         num_disk_as_reward=args.num_disk_as_reward,
         render=args.render)
