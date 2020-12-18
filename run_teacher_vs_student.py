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
import torch.multiprocessing as mp
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass


def play(teacher,
         teacher_agent_type='rainbow',
         student_agent_type='rainbow',
         opponent_agent_type='',
         board_size=8,
         num_rounds=100,
         teacher_search_depth=1,
         student_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         test_init_rand_steps=10,
         num_disk_as_reward=True,
         render=False,
         train_teacher=False,
         test_interval=2500,
         num_test_games=200,
         teacher_train_steps=5000,
         save_interval=5000,
         # load_path='',
         # load_path='data/selfplay/rainbow_selfplay_350000.pth',
         # load_path='/data/unagi0/omura/othello/selfplay/rainbow_selfplay_350000.pth',
         # load_path='/data/unagi0/omura/othello/selfplay/rainbow_selfplay_2nd_65000.pth',
         load_path='/data/unagi0/omura/othello/teacher_student/rainbow_gre_rand_teacher_train_10interval_mp_59999.pth',
         num_process=1
         ):
    print('teacher: {}'.format(teacher_agent_type))
    print('student: {}'.format(student_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    agent_name_teacher = 'rainbow_gre_rand_teacher_notrain_mp_load_teacher60k'
    agent_name_student = 'rainbow_gre_rand_student_notrain_mp_load_teacher60k'
    # agent_name_teacher = 'test'
    # agent_name_student = 'test'
    # load_path = ''

    teacher_policy = create_policy(
        policy_type=teacher_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=teacher_search_depth,
        agent_name=agent_name_teacher
    )
    student_policy = create_policy(
        policy_type=student_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=student_search_depth,
        agent_name=agent_name_student)

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
    # opponent_policies = [('greedy', opponent_policy1)]

    if not train_teacher:
        def noop(*args):
            pass
        teacher_policy.run = noop
    # if not hasattr(protagonist_policy, 'run'):
    #     protagonist_policy.run = nop

    # if opponent_agent_type == 'human':
    #     render_in_step = True
    # else:
    #     render_in_step = False

    if load_path:
        print('Load {} ...'.format(load_path))
        start_episode, loss = load(teacher_policy, load_path)
    else:
        start_episode = 0

    env = othello.SimpleOthelloEnv(
            board_size=board_size,
            seed=rand_seed,
            initial_rand_steps=env_init_rand_steps,
            num_disk_as_reward=num_disk_as_reward,
            render_in_step=render)
    #
    # env_test = othello.OthelloEnv(
    #     board_size=board_size,
    #     seed=rand_seed,
    #     initial_rand_steps=env_init_rand_steps,
    #     num_disk_as_reward=num_disk_as_reward,
    #     render_in_step=render)

    win_cnts = draw_cnts = lose_cnts = 0
    win_per = {'rand': 0, 'greedy': 0}
    last_win_per = {'rand': 0, 'greedy': 0}
    teacher_queue = queue.Queue()
    # for i in range(start_episode, num_rounds):
    for i in range(num_rounds):
        switch = np.random.randint(2)
        if switch:
            teacher = teacher * -1

        policy = {}
        if teacher == -1:
            tcolor = 'black'
            policy['black'] = teacher_policy
            policy['white'] = student_policy
        else:
            tcolor = 'white'
            policy['black'] = student_policy
            policy['white'] = teacher_policy

        print('Episode {}'.format(i + 1))
        print('Teacher is {}'.format(tcolor))

        def run(color, state, action, reward, done, next_state):
            if color == tcolor:
                if done:
                    teacher_reward = 0
                    for k in win_per.keys():
                        teacher_reward += win_per[k] - last_win_per[k]
                else:
                    teacher_reward = 0
                if student_policy.is_learning():
                    # print('### learning')
                    teacher_queue.put((state, action, teacher_reward, done, next_state))

            else:
                policy[color].run(state, action, reward, done, next_state)

        obs_b = env.reset()
        state_b = make_state(obs_b, env)
        teacher_policy.reset(env)
        student_policy.reset(env)
        if render:
            env.render()
        done_b = done_w = False
        init = True

        # student_policy.win_queue = mp.Queue(num_process)

        while not (done_b or done_w):
            # black
            assert env.player_turn == -1
            action_b = policy['black'].get_action(state_b)
            next_obs_b, reward_b, done_b, _ = env.step(action_b)
            next_state_b = make_state(next_obs_b, env)
            while (not done_b) and env.player_turn == -1:
                # policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                run('black', state_b, action_b, reward_b, done_b, next_state_b)
                action_b = policy['black'].get_action(next_state_b)
                next_obs_b, reward_b, done_b, _ = env.step(action_b)
                next_state_b = make_state(next_obs_b, env)

            # learning black policy
            if not init:
                # policy['white'].run(state_w, action_w, - reward_b, done_b, next_state_b)
                run('white', state_w, action_w, - reward_b, done_b, next_state_b)
            init = False
            if done_b:
                # policy['black'].run(state_b, action_b, reward_b, done_b, next_state_b)
                run('black', state_b, action_b, reward_b, done_b, next_state_b)
                break

            # white
            assert env.player_turn == 1
            state_w = next_state_b
            action_w = policy['white'].get_action(state_w)
            next_obs_w, reward_w, done_w, _ = env.step(action_w)
            next_state_w = make_state(next_obs_w, env)
            while (not done_w) and env.player_turn == 1:
                # policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                run('white', state_w, action_w, reward_w, done_w, next_state_w)
                action_w = policy['white'].get_action(next_state_w)
                next_obs_w, reward_w, done_w, _ = env.step(action_w)
                next_state_w = make_state(next_obs_w, env)

            # learning black policy
            # policy['black'].run(state_b, action_b, - reward_w, done_w, next_state_w)
            run('black', state_b, action_b, - reward_w, done_w, next_state_w)
            if done_w:
                # policy['white'].run(state_w, action_w, reward_w, done_w, next_state_w)
                run('white', state_w, action_w, reward_w, done_w, next_state_w)
                break

            state_b = next_state_w

            if render:
                env.render()

        if done_w:
            reward = reward_w * teacher
        elif done_b:
            reward = reward_b * -teacher
        else:
            raise ValueError

        print('reward={}'.format(reward))
        if num_disk_as_reward:
            total_disks = board_size ** 2
            if teacher == 1:
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

        if teacher_queue.qsize() >= teacher_train_steps:
            while not teacher_queue.empty():
                trans = teacher_queue.get()
                teacher_policy.run(*trans)

        # calc student's winning %
        args_student = (student_agent_type, board_size, rand_seed, student_search_depth, 'calc_win_rate')

        state_dict = student_policy.network_state_dict()
        if i % test_interval == 0:
            env.initial_rand_steps = test_init_rand_steps
            for name, opponent_policy in opponent_policies:
                win_queue = mp.Queue(num_process)
                p_games = num_test_games // num_process
                total_games = p_games * num_process
                ps = []
                student_policies = []
                # for j in range(num_process):
                #     student_policies.append(copy.deepcopy(student_policy))

                for j in range(num_process):
                    ps.append(mp.Process(target=calc_win, args=(env, p_games, args_student, state_dict, opponent_policy, win_queue)))
                for p in ps:
                    p.start()
                    # time.sleep(0.5)
                for p in ps:
                    p.join()

                # assert win_queue.qsize() == num_process

                total_wins = 0
                for _ in range(num_process):
                    total_wins += win_queue.get()

                last_win_per[name] = win_per[name]
                win_per[name] = total_wins / total_games
                student_policy.writer.add_scalar("win%({})".format(name), win_per[name], i)
            print()
            print('last win%:', last_win_per)
            print('win%:', win_per)
            print()
            env.initial_rand_steps = env_init_rand_steps

        if (i+1) % save_interval == 0:
            teacher_path = '/data/unagi0/omura/othello/teacher_student/{}_{}.pth'.format(agent_name_teacher, i+1)
            student_path = '/data/unagi0/omura/othello/teacher_student/{}_{}.pth'.format(agent_name_student, i+1)
            # teacher_path = 'data/teacher_student/{}_{}.pth'.format(agent_name_teacher, i)
            # student_path = 'data/teacher_student/{}_{}.pth'.format(agent_name_student, i)
            save(i, teacher_policy, 0, teacher_path)
            save(i, student_policy, 0, student_path)

    env.close()


def calc_win(env, num_test_games, args_policy, state_dict, opponent_policy, win_queue):
    env = copy.deepcopy(env)
    student_policy = create_policy(*args_policy)
    student_policy.load_state_dict(state_dict)
    wins = 0
    for j in range(num_test_games):
        student = np.random.randint(2)
        student = -1 if student == 0 else 1
        policy = {}
        if student == -1:
            policy['black'] = student_policy
            policy['white'] = opponent_policy
        else:
            policy['black'] = opponent_policy
            policy['white'] = student_policy

        obs_b = env.reset()
        state_b = make_state(obs_b, env)
        student_policy.reset(env)
        opponent_policy.reset(env)

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
            reward = reward_w * student
        elif done_b:
            reward = reward_b * -student
        else:
            raise ValueError
        if reward > 0:
            wins += 1

    print('### games', num_test_games, 'wins', wins)
    win_queue.put(wins)


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo', 'rainbow'])
    parser.add_argument('--student', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo', 'rainbow'])
    parser.add_argument('--opponent', default='rand',
                        choices=['rand', 'greedy', 'maximin', 'human', 'dqn', 'ppo', 'rainbow'])
    parser.add_argument('--teacher-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--teacher-search-depth', default=1, type=int)
    parser.add_argument('--student-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=100, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    args, _ = parser.parse_known_args()

    # Run test plays.
    teacher = 1 if args.teacher_plays_white else -1
    teacher_agent_type = args.teacher
    student_agent_type = args.student
    opponent_agent_type = args.opponent
    play(teacher=teacher,
         teacher_agent_type=teacher_agent_type,
         student_agent_type=student_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         teacher_search_depth=args.teacher_search_depth,
         student_search_depth=args.student_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         num_disk_as_reward=args.num_disk_as_reward,
         render=not args.no_render)

