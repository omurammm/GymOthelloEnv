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
import copy

from torch.utils.tensorboard import SummaryWriter

from envs import PPOTeacherStudentEnvs

from multiprocessing import set_start_method
#
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass


def test(protagonist,
         protagonist_agent_type,
         opponent_agent_type,
         board_size,
         num_rounds,
         protagonist_search_depth,
         opponent_search_depth,
         rand_seed,
         env_init_rand_steps,
         num_disk_as_reward=True,
         test_init_rand_steps=10,
         render=False,
         train_teacher=True,
         test_interval=10,
         num_test_games=200,
         save_interval=500,
         # load_path='data/selfplay/ent0_lr1e-5_35000.pth'):
         # load_path='/data/unagi0/omura/othello/selfplay/ent0_lr1e-5_35000.pth'):
         load_path = '/data/unagi0/omura/othello/teacher_student/testinterval10_ent0_lr5e-6_clip1e-1_numstep64_teacher_10000.pth'):

    args = get_args()
    args.algo = 'ppo'
    args.use_gae = True
    args.lr = 5e-6 #2.5e-4
    args.clip_param = 0.1
    args.value_loss_coef = 0.5 #0.5
    args.num_processes = 8
    args.num_steps = 64
    args.num_mini_batch = 4
    args.log_interval = 1
    args.use_linear_lr_decay = True
    args.entropy_coef = 0.0 # 0.01
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
    # agent_name = 'wo_ttrain_ent0_lr5e-6_clip1e-1_numstep64'
    # agent_name = 'testinterval10_ent0_lr5e-6_clip1e-1_numstep64'
    # agent_name = 'trained1_10k_wo_ttrain_ent0_lr5e-6_clip1e-1_numstep64'
    agent_name = 'trained1_10k_testinterval10_ent0_lr5e-6_clip1e-1_numstep64'
    # agent_name = 'test'
    writer = SummaryWriter(log_dir="./log/ppo_teacher_vs_student/{}".format(agent_name))

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
        actor_critic_teacher = torch.load(load_path)
    else:
        actor_critic_teacher = Policy(
            obs_space.shape,
            action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_student = Policy(
        obs_space.shape,
        action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic_student.to(device)
    actor_critic_teacher.to(device)

    envs = PPOTeacherStudentEnvs(envs_list, othello_teacher_vs_student, actor_critic_teacher, actor_critic_student, device)
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
        agent_teacher = algo.PPO(
            actor_critic_teacher,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
        agent_student = algo.PPO(
            actor_critic_student,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent_teacher = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)
        agent_student = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts_teacher = RolloutStorage(args.num_steps, args.num_processes,
                              obs_space.shape, action_space,
                              actor_critic_teacher.recurrent_hidden_state_size)
    rollouts_student = RolloutStorage(args.num_steps, args.num_processes,
                              obs_space.shape, action_space,
                              actor_critic_student.recurrent_hidden_state_size)

    # episode_rewards = deque(maxlen=10)
    update_t = 0
    update_s = 0
    win_avg = {'rand': 0, 'greedy': 0}
    last_win_avg = {'rand': 0, 'greedy': 0}

    obs_ts = [[0]*args.num_processes, [0]*args.num_processes]
    action_ts = [[0] * args.num_processes, [0] * args.num_processes]
    reward_ts = [[0] * args.num_processes, [0] * args.num_processes]
    done_ts = [[0] * args.num_processes, [0] * args.num_processes]
    infos_ts = [[0] * args.num_processes, [0] * args.num_processes]
    v_logprob_hidden_ts = [[0] * args.num_processes, [0] * args.num_processes]
    masks_ts = [[0] * args.num_processes, [0] * args.num_processes]
    bad_masks_ts = [[0] * args.num_processes, [0] * args.num_processes]
    choices_ts = [[0] * args.num_processes, [0] * args.num_processes]

    def save_to_buffer(who_, idx, obs_, action_, reward_, done_, infos_, v_logprob_hidden_, masks_, bad_masks_, choices_):
        if who_ == 'teacher':
            ts = 0
        else:
            ts = 1
        obs_ts[ts][idx] = obs_[idx]
        action_ts[ts][idx] = action_[idx]
        reward_ts[ts][idx] = reward_[idx]
        done_ts[ts][idx] = done_[idx]
        infos_ts[ts][idx] = infos_[idx]
        v_logprob_hidden_ts[ts][idx] = v_logprob_hidden_[idx]
        masks_ts[ts][idx] = masks_[idx]
        bad_masks_ts[ts][idx] = bad_masks_[idx]
        choices_ts[ts][idx] = choices_[idx]

    student_buffer = {}
    teacher_buffer = {}
    for i in range(args.num_processes):
        student_buffer[i] = 0
        teacher_buffer[i] = 0

    teacher_step = 0
    student_step = 0
    for episode in range(num_rounds):
        print()
        print('Episode %s' % episode)
        teacher = random.choice([1, -1])
        envs.reset(teacher, win_avg, last_win_avg)
        over = False
        done_ts = [[0] * args.num_processes, [0] * args.num_processes]
        # teacher_step = 0
        # student_step = 0
        accum_reward_s = np.zeros(args.num_processes)
        accum_reward_t = np.zeros(args.num_processes)
        while not over:
            over = all(np.array(done_ts[0])+np.array(done_ts[1]))

            # Observe reward and next obs
            # if not over:
            t_or_s, obs, action, reward, done, infos, v_logprob_hidden, masks, bad_masks = envs.step(rollouts_student.recurrent_hidden_states[student_step % args.num_steps],
                                                                                             rollouts_teacher.recurrent_hidden_states[teacher_step % args.num_steps])

            # print('@', over, t_or_s, done, reward.squeeze())
            # print(action.squeeze())
            choices = [info['choices'] for info in infos]
            # for i in range(len(action)):
            #     assert done[i] or action[i][0] in choices[i], (action[i][0], choices[i])

            for i, who in enumerate(t_or_s):
                save_to_buffer(who, i, obs, action, reward, done, infos, v_logprob_hidden, masks, bad_masks, choices)
                if who == 'teacher':
                    teacher_buffer[i] = 1
                else:
                    student_buffer[i] = 1

            # print(action_ts, choices_ts)

            if all(list(teacher_buffer.values())) or over:
                obs_t = torch.stack(obs_ts[0])
                action_t = torch.stack(action_ts[0])
                reward_t = torch.stack(reward_ts[0])
                # done_t = done_ts[0]
                # infos_t = infos_ts[0]
                v_logprob_hidden_t = torch.stack(v_logprob_hidden_ts[0])
                masks_t = torch.stack(masks_ts[0])
                bad_masks_t = torch.stack(bad_masks_ts[0])
                choices_t = copy.deepcopy(choices_ts[0])

                accum_reward_t = accum_reward_t + np.array(reward_t.squeeze())
                # print('t', accum_reward_t, np.array(reward_t.squeeze()))

                if teacher_step == 0:
                    rollouts_teacher.obs[0].copy_(obs_t)
                    rollouts_teacher.masks[0].copy_(masks_t)
                    rollouts_teacher.bad_masks[0].copy_(bad_masks_t)
                else:
                    rollouts_teacher.insert(obs_t, prev_hidden_t, prev_action_t,
                                            prev_logprob_t, prev_value_t, prev_reward_t, masks_t, bad_masks_t,
                                            prev_choices_t)
                prev_action_t = action_t
                prev_value_t = v_logprob_hidden_t[:, 0].unsqueeze(1)
                prev_logprob_t = v_logprob_hidden_t[:, 1].unsqueeze(1)
                prev_hidden_t = v_logprob_hidden_t[:, 2].unsqueeze(1)
                prev_reward_t = reward_t
                # prev_masks = masks
                # prev_bad_masks = bad_masks
                prev_choices_t = copy.deepcopy(choices_t)
                # over_t = all(done_t)
                teacher_step += 1
                for i in range(args.num_processes):
                    teacher_buffer[i] = 0

            if all(list(student_buffer.values())) or over:
                obs_s = torch.stack(obs_ts[1])
                action_s = torch.stack(action_ts[1])
                reward_s = torch.stack(reward_ts[1])
                # done_s = done_ts[1]
                # infos_s = infos_ts[1]
                v_logprob_hidden_s = torch.stack(v_logprob_hidden_ts[1])
                masks_s = torch.stack(masks_ts[1])
                bad_masks_s = torch.stack(bad_masks_ts[1])
                choices_s = copy.deepcopy(choices_ts[1])

                accum_reward_s = accum_reward_s + np.array(reward_s.squeeze())
                # print('s', accum_reward_s, np.array(reward_s.squeeze()))

                if student_step == 0:
                    rollouts_student.obs[0].copy_(obs_s)
                    rollouts_student.masks[0].copy_(masks_s)
                    rollouts_student.bad_masks[0].copy_(bad_masks_s)
                else:
                    rollouts_student.insert(obs_s, prev_hidden_s, prev_action_s,
                                            prev_logprob_s, prev_value_s, prev_reward_s, masks_s, bad_masks_s,
                                            prev_choices_s)
                prev_action_s = action_s
                prev_value_s = v_logprob_hidden_s[:, 0].unsqueeze(1)
                prev_logprob_s = v_logprob_hidden_s[:, 1].unsqueeze(1)
                prev_hidden_s = v_logprob_hidden_s[:, 2].unsqueeze(1)
                prev_reward_s = reward_s
                # prev_masks = masks
                # prev_bad_masks = bad_masks
                prev_choices_s = copy.deepcopy(choices_s)
                # over_s = all(done_s)
                student_step += 1
                for i in range(args.num_processes):
                    student_buffer[i] = 0

            if (teacher_step % args.num_steps == 0) and (teacher_step != 0):
                if train_teacher:
                    with torch.no_grad():
                        next_value_teacher = actor_critic_teacher.get_value(
                            rollouts_teacher.obs[-1], rollouts_teacher.recurrent_hidden_states[-1],
                            rollouts_teacher.masks[-1]).detach()
                    rollouts_teacher.compute_returns(next_value_teacher, args.use_gae, args.gamma,
                                             args.gae_lambda, args.use_proper_time_limits)
                    value_loss_teacher, action_loss_teacher, dist_entropy_teacher = agent_teacher.update(
                        rollouts_teacher)
                    rollouts_teacher.after_update()
                    if args.use_linear_lr_decay:
                        utils.update_linear_schedule(
                            agent_teacher.optimizer, update_t, num_updates,
                            agent_teacher.optimizer.lr if args.algo == "acktr" else args.lr)
                    update_t += 1
                # teacher_step = 0

            if (student_step % args.num_steps == 0) and (student_step != 0):
                with torch.no_grad():
                    next_value_student = actor_critic_student.get_value(
                        rollouts_student.obs[-1], rollouts_student.recurrent_hidden_states[-1],
                        rollouts_student.masks[-1]).detach()
                rollouts_student.compute_returns(next_value_student, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)
                value_loss_student, action_loss_student, dist_entropy_student = agent_student.update(
                    rollouts_student)
                rollouts_student.after_update()
                if args.use_linear_lr_decay:
                    utils.update_linear_schedule(
                        agent_student.optimizer, update_s, num_updates,
                        agent_student.optimizer.lr if args.algo == "acktr" else args.lr)
                update_s += 1
                # student_step = 0

            if over:
                student_wins = 0
                print('reward')
                print(accum_reward_s)
                for r in accum_reward_s:
                    if r > 0:
                        student_wins += 1
                student_win_percent = student_wins / len(accum_reward_s)
            # over = all(done_ts[0]) and all(done_ts[1])
            # over = all(np.array(done_ts[0])+np.array(done_ts[1]))


        if episode % test_interval == 0:
            print('Test')
            games_rand, wins_rand = envs.test('rand', num_test_games, rollouts_student.recurrent_hidden_states[0])
            writer.add_scalar("win avg({})".format('rand'), wins_rand/games_rand, episode)
            print('### vs-random winning% {}/{}={}'.format(wins_rand, games_rand, wins_rand / games_rand))
            games_greedy, wins_greedy = envs.test('greedy', num_test_games, rollouts_student.recurrent_hidden_states[0])
            writer.add_scalar("win avg({})".format('greedy'), wins_greedy/games_greedy, episode)
            print('### vs-greedy winning% {}/{}={}'.format(wins_greedy, games_greedy, wins_greedy/games_greedy))
            last_win_avg = copy.deepcopy(win_avg)
            win_avg['rand'] = wins_rand / games_rand
            win_avg['greedy'] = wins_greedy / games_greedy
        if episode % save_interval == 0:
            if os.path.exists('/data/unagi0/omura'):
                t_save_path = '/data/unagi0/omura/othello/teacher_student/{}_teacher_{}.pth'.format(agent_name, episode)
                s_save_path = '/data/unagi0/omura/othello/teacher_student/{}_student_{}.pth'.format(agent_name, episode)
            else:
                t_save_path = 'data/selfplay/{}_teacher_{}.pth'.format(agent_name, episode)
                s_save_path = 'data/selfplay/{}_student_{}.pth'.format(agent_name, episode)
            torch.save(actor_critic_teacher, t_save_path)
            torch.save(actor_critic_student, s_save_path)

        if teacher_step > args.num_steps and student_step > args.num_steps:
            writer.add_scalar("value_loss_student", value_loss_student, episode)
            writer.add_scalar("action_loss_student", action_loss_student, episode)
            writer.add_scalar("dist_entropy_student", dist_entropy_student, episode)
            writer.add_scalar("student_win_percent", student_win_percent, episode)
            if train_teacher:
                writer.add_scalar("value_loss_teacher", value_loss_teacher, episode)
                writer.add_scalar("action_loss_teacher", action_loss_teacher, episode)
                writer.add_scalar("dist_entropy_teacher", dist_entropy_teacher, episode)
        # print(value_loss, action_loss, dist_entropy)

    envs.over()


def random_possible_actions(infos):
    actions = []
    for info in infos:
        if len(info['choices']) != 0:
            actions.append(random.choice(info['choices']))
        else:
            actions.append(-1)
    return actions


def othello_teacher_vs_student(id, env, pipe, parent_pipe):
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
            print('##', id, cmd, done)
        else:
            recv = True

        if cmd == 'over':
            break
        elif cmd == 'reset':
            obs_b = env.reset()
            state_b = make_state(obs_b, env)
            # protagonist = np.random.randint(2)
            teacher, win_avg, last_win_avg = a
            tcolor = 'black' if teacher == -1 else 'white'
            int_to_str = {-1: 'black', 1: 'white'}
            done = False
            done_b = done_w = False
            init = True

        elif cmd == 'step':
            def send_transition(color, state, action, reward, done_, choices, output):
                if color == tcolor:
                    if done_:
                        teacher_reward = 0
                        for k in win_avg.keys():
                            teacher_reward += win_avg[k] - last_win_avg[k]
                    else:
                        teacher_reward = 0
                    # teacher_queue.put((state, action, teacher_reward, done, next_state))
                    pipe.send((state, action, teacher_reward, done_,
                                   {'type': '{}_transition'.format(int_to_str[teacher]), 'choices': choices}, output))
                    recv_ = pipe.recv()[0]

                else:
                    # policy[color].run(state, action, reward, done, next_state)
                    pipe.send((state, action, reward, done_,
                               {'type': '{}_transition'.format(int_to_str[teacher*-1]), 'choices': choices}, output))
                    recv_ = pipe.recv()[0]
                return recv_

            if done:
                pipe.send((o, 0, 0, done, {'type': 'over', 'choices': []}, dummy_outputs))
                cmd, a = pipe.recv()
                if cmd == 'reset':
                    recv = False
                elif cmd == 'over':
                    break
                continue

            while not (done_b or done_w):
                # black
                assert env.player_turn == -1
                pipe.send((state_b, 0, 0, 0, {'type': 'need_black_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_b, output_b = pipe.recv()
                choice_b = env.possible_moves
                assert cmd == 'step'
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)
                while (not done_b) and env.player_turn == -1:
                    cmd = send_transition('black', state_b, action_b, reward_b, done_b, choice_b, output_b)
                        # pipe.send((state_b, action_b, reward_b, done_b, {'type': 'transition', 'choices': choice_b}, output_b))
                        # cmd = pipe.recv()[0]
                    assert cmd == 'step'
                    pipe.send((next_state_b, 0, 0, 0, {'type': 'need_black_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_b, output_b = pipe.recv()
                    choice_b = env.possible_moves
                    assert cmd == 'step'
                    next_obs_b, reward_b, done_b, _ = env.step(action_b)
                    next_state_b = make_state(next_obs_b, env)

                if not init:
                    # if pcolor == 'white':
                    cmd = send_transition('white', state_w, action_w, - reward_b, done_b, choice_w, output_w)
                        # pipe.send((state_w, action_w, - reward_b, done_b, {'type': 'transition', 'choices': choice_w}, output_w))
                        # cmd = pipe.recv()[0]
                    assert cmd == 'step'

                if done_b:
                    cmd = send_transition('black', state_b, action_b, reward_b, done_b, choice_b, output_b)
                    # pipe.send((state_b, action_b, reward_b, done_b, {'type': 'transition', 'choices': env.possible_moves}, output_b))
                    # cmd = pipe.recv()[0]
                    assert cmd == 'step'
                    # if init:
                    # cmd = send_transition('white', o, 0, 0, done_b, [], dummy_outputs)
                        # pipe.send((o, 0, 0, done_b, {'type': 'transition', 'choices': env.possible_moves}, dummy_outputs))
                        # cmd = pipe.recv()[0]
                    # assert cmd == 'step'
                    break
                init = False

                # white
                assert env.player_turn == 1
                state_w = next_state_b
                pipe.send((state_w, 0, 0, 0, {'type': 'need_white_action', 'choices': env.possible_moves}, dummy_outputs))
                cmd, action_w, output_w = pipe.recv()
                choice_w = env.possible_moves
                assert cmd == 'step'
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env)
                while (not done_w) and env.player_turn == 1:
                    cmd = send_transition('white', state_w, action_w, reward_w, done_w, choice_w, output_w)
                        # pipe.send((state_w, action_w, reward_w, done_w, {'type': 'transition', 'choices': choice_w}, output_w))
                        # cmd = pipe.recv()[0]
                    assert cmd == 'step'
                    pipe.send((next_state_w, 0, 0, 0, {'type': 'need_white_action', 'choices': env.possible_moves}, dummy_outputs))
                    cmd, action_w, output_w = pipe.recv()
                    choice_w = env.possible_moves
                    assert cmd == 'step'
                    next_obs_w, reward_w, done_w, _ = env.step(action_w)
                    next_state_w = make_state(next_obs_w, env)

                # learning black policy
                cmd = send_transition('black', state_b, action_b, - reward_w, done_w, choice_b, output_b)
                    # pipe.send((state_b, action_b, - reward_w, done_w, {'type': 'transition', 'choices': choice_b}, output_b))
                    # cmd, pipe.recv()[0]
                assert cmd == 'step'
                if done_w:
                    cmd = send_transition('white', state_w, action_w, reward_w, done_w, choice_w, output_w)
                        # pipe.send((state_w, action_w, reward_w, done_w, {'type': None, 'choices': choice_w}, output_w))
                        # cmd = pipe.recv()[0]
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
    parser.add_argument('--num-disk-as-reward', default=True,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=50000, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
    parser.add_argument('--render', default=False, action='store_true')
    args, _ = parser.parse_known_args()
    print(args)
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
