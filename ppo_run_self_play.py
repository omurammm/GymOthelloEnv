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




def test(protagonist,
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

    args = get_args()
    args.algo = 'ppo'
    args.use_gae = True
    args.lr = 2.5e-4
    args.clip_param = 0.1
    args.value_loss_coef = 0.5
    args.num_processes = 2 #8
    args.num_steps = 8 #128
    args.num_mini_batch = 4
    args.log_interval = 1
    args.use_linear_lr_decay = True
    args.entropy_coef = 0.01
    print(args)
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
    device = torch.device("cuda:0" if args.cuda else "cpu")

    agent_name = 'test'

    num_process = 2
    num_test_games = 20
    envs_list = []
    for _ in range(num_process):
        env = othello.SimpleOthelloEnv(
            board_size=board_size,
            seed=rand_seed,
            initial_rand_steps=env_init_rand_steps,
            num_disk_as_reward=num_disk_as_reward,
            render_in_step=render)
        envs_list.append(env)
    # protagonist_policy = create_policy(
    #     policy_type=protagonist_agent_type,
    #     board_size=board_size,
    #     seed=rand_seed,
    #     search_depth=protagonist_search_depth,
    #     agent_name=agent_name)


    obs_space = spaces.Box(np.zeros((4, 8, 8)), np.ones((4, 8, 8)))
    action_space = spaces.Discrete(board_size ** 2)

    actor_critic = Policy(
        obs_space.shape,
        action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    envs = Envs(envs_list, actor_critic, device)

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

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              obs_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    num_episode = 100
    step_per_episode = 32
    num_updates = (num_episode * step_per_episode) // args.num_steps // args.num_processes
    num_updates = 30

    episode_rewards = deque(maxlen=10)

    u = 0
    for i in range(num_episode):
        print()
        print('Episode %s' % i)
        # obs, info = envs.reset()
        # rollouts.obs[0].copy_(obs)
        # rollouts.to(device)
        #TODO mask
        envs.reset()

        over = False
        while not over:
            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, u, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):

                # Obser reward and next obs
                if not over:
                    obs, action, reward, done, infos, v_logprob_hidden, masks, bad_masks = envs.step(rollouts.recurrent_hidden_states[step])

                # for info in infos:
                #     if 'episode' in info.keys():
                #         episode_rewards.append(info['episode']['r'])

                if step == 0:
                    rollouts.obs[0].copy_(obs)
                    rollouts.masks[0].copy_(masks)
                    rollouts.bad_masks[0].copy_(bad_masks)
                else:
                    rollouts.insert(obs, prev_hidden, prev_action,
                                    prev_logprob, prev_value, prev_reward, masks, bad_masks)
                # prev_obs = obs
                prev_action = action
                prev_value = v_logprob_hidden[:, 0].unsqueeze(1)
                prev_logprob = v_logprob_hidden[:, 1].unsqueeze(1)
                prev_hidden = v_logprob_hidden[:, 2].unsqueeze(1)
                prev_reward = reward
                # prev_masks = masks
                # prev_bad_masks = bad_masks
                over = all(done)
            print('steppppppppp')
            print()
            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

        games, wins = envs.test('rand', num_test_games, rollouts.recurrent_hidden_states[0])
        print('### winning% {}/{}={}'.format(wins, games, wins/games))

    envs.over()





class Envs():
    def __init__(self, envs, policy, device):
        self.envs = envs
        self.policy = policy
        self.device = device
        self.procs = []
        self.parent_pipes = []
        board_size = envs[0].board_size
        self.observation_space = spaces.Box(np.zeros((4, board_size, board_size)), np.ones((4, board_size, board_size)))
        self.action_space = spaces.Discrete(board_size ** 2)

        # self.obs_shape = (4, 8, 8)
        # self.action_shape = 64

        for i, env in enumerate(envs):
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(target=subproc_worker, args=(i, env, child_pipe, parent_pipe))
            self.procs.append(proc)
            self.parent_pipes.append(parent_pipe)
            proc.start()
            child_pipe.close()

    def over(self):
        for pipe in self.parent_pipes:
            pipe.send(('over', None))

    def reset(self):
        for pipe in self.parent_pipes:
            pipe.send(('reset', None))
        for pipe in self.parent_pipes:
            pipe.send(('step', -3))
        self.is_reset = True

    def step(self, ex_hiddens):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, actions, rewards, dones, infos, outputs = zip(*outs)
        obs, actions, rewards, dones, infos, outputs = list(obs), list(actions), list(rewards), list(dones), list(infos), list(outputs)
        # print(obs)
        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device).unsqueeze(1)
        outputs = torch.from_numpy(np.array(outputs)).float().to(self.device)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        while 'need_action' in [info['type'] for info in infos]:
            choices = [info['choices'] for info in infos]
            with torch.no_grad():
                v, next_actions, logprob, hidden = self.policy.act(obs, ex_hiddens, masks, choices)
            # next_actions = random_possible_actions(infos)
            for i, info in enumerate(infos):
                if info['type'] != 'need_action':
                    continue
                self.parent_pipes[i].send(('step', int(next_actions[i][0]), (v[i], logprob[i], hidden[i])))
                out = self.parent_pipes[i].recv()
                # print('$$ info', out[-1])
                obs[i] = torch.from_numpy(out[0])
                actions[i] = torch.from_numpy(np.array([out[1]]))
                rewards[i] = torch.from_numpy(np.array([out[2]]))
                dones[i] = out[3]
                infos[i] = out[4]
                outputs[i] = torch.from_numpy(np.array(out[5]))
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        for pipe in self.parent_pipes:
            pipe.send(('step', -1))

        if not all(dones):
            for i, d in enumerate(dones):
                if d:
                    self.parent_pipes[i].send(('step', -2))
                    # _ = self.parent_pipes[i].recv()

        return obs, actions, rewards, dones, infos, outputs, masks, bad_masks

    def test(self, name, num_games, ex_hiddens):
        proc_num_games = num_games // len(self.parent_pipes)
        for pipe in self.parent_pipes:
            pipe.send(('test-'+name, proc_num_games))

        over = False
        while not over:
            outs = [pipe.recv() for pipe in self.parent_pipes]
            obs, actions, rewards, dones, infos, outputs = zip(*outs)
            obs, actions, rewards, dones, infos, outputs = list(obs), list(actions), list(rewards), list(
                dones), list(
                infos), list(outputs)

            over = all([info['type'] == 'over' for info in infos])
            if over:
                break
            obs = torch.from_numpy(np.array(obs)).float().to(self.device)
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
            choices = [info['choices'] for info in infos]
            with torch.no_grad():
                v, next_actions, logprob, hidden = self.policy.act(obs, ex_hiddens, masks, choices)
            # next_actions = random_possible_actions(infos)
            for i, info in enumerate(infos):
                self.parent_pipes[i].send(('step', int(next_actions[i][0]), (v[i], logprob[i], hidden[i])))
        for pipe in self.parent_pipes:
            pipe.send(('finish-test', None, None))
        num_wins = [info['wins'] for info in infos]
        print(num_wins)
        return proc_num_games*len(self.parent_pipes), sum(num_wins)



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
    r = 0
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

        print('### first subp({}) action, cmd:'.format(id), a, cmd)
        if cmd == 'over':
            break
        elif cmd == 'reset':
            print('### subp({}) reset'.format(id))
            obs_b = env.reset()
            state_b = make_state(obs_b, env)
            protagonist = np.random.randint(2)
            protagonist = -1 if protagonist == 0 else 1
            pcolor = 'black' if protagonist == -1 else 'white'
            print('### pcolor %s' % pcolor)
            done = False
            done_b = done_w = False
            init = True
            # pipe.send((state_b, -1, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}))
            # pipe.send((state_b, {'type': 'reset', 'choices': env.possible_moves}))

        elif cmd == 'step':
            print('# subp({}) start action:'.format(id), a)

            if done:
                print('### done', id)
                pipe.send((o, 0, 0, done, {'type': 'over', 'choices': env.possible_moves}, dummy_outputs))
                cmd, a = pipe.recv()
                # assert cmd == 'step', 'cmd={}'.format(cmd)
                if cmd == 'reset':
                    recv = False
                elif cmd == 'over':
                    break
                continue

            # protagonist_policy.reset(env)

            while not (done_b or done_w):
                # black
                assert env.player_turn == -1
                pipe.send((state_b, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_b, output_b = pipe.recv()
                assert cmd == 'step'
                # action_b = action('black', pcolor, state_b, policy)
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)
                while (not done_b) and env.player_turn == -1:
                    if pcolor == 'black':
                        pipe.send((state_b, action_b, reward_b, done_b, {'type': None, 'choices': env.possible_moves}, output_b))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                        # policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                    pipe.send((next_state_b, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_b, output_b = pipe.recv()
                    assert cmd == 'step'
                    next_obs_b, reward_b, done_b, _ = env.step(action_b)
                    next_state_b = make_state(next_obs_b, env)

                # learning black policy
                if not init:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, - reward_b, done_b, {'type': None, 'choices': env.possible_moves}, output_w))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'

                if done_b:
                    if pcolor == 'black':
                        pipe.send((state_b, action_b, reward_b, done_b, {'type': None, 'choices': env.possible_moves}, output_b))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    if pcolor == 'white' and init:
                        pipe.send(([0], 0, 0, done_b, {'type': None, 'choices': env.possible_moves}, dummy_outputs))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    break
                init = False

                # white
                assert env.player_turn == 1
                state_w = next_state_b
                pipe.send((state_w, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_w, output_w = pipe.recv()
                assert cmd == 'step'
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env)
                while (not done_w) and env.player_turn == 1:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, reward_w, done_w, {'type': None, 'choices': env.possible_moves}, output_w))
                        cmd = pipe.recv()[0]
                        assert cmd == 'step'
                    pipe.send((next_state_w, 0, 0, 0, {'type': 'need_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_w, output_w = pipe.recv()
                    assert cmd == 'step'
                    next_obs_w, reward_w, done_w, _ = env.step(action_w)
                    next_state_w = make_state(next_obs_w, env)

                # learning black policy
                if pcolor == 'black':
                    pipe.send((state_b, action_b, - reward_w, done_w, {'type': None, 'choices': env.possible_moves}, output_b))
                    cmd, pipe.recv()[0]
                    assert cmd == 'step'
                if done_w:
                    if pcolor == 'white':
                        pipe.send((state_w, action_w, reward_w, done_w, {'type': None, 'choices': env.possible_moves}, output_w))
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

def rule_base_game(name, num_games, env, pipe):
    o = np.zeros((4, env.board_size, env.board_size))
    r = 0
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
    return num_wins




# def action(color, p_color, state, policy):
#     if color == p_color:
#         action = policy[color].get_action(state)
#     else:
#         action = policy[color].get_test_action(state)
#     return action


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
