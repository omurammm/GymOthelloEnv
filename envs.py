import torch
import multiprocessing as mp
import random
from gym import spaces
import numpy as np

class Envs():
    def __init__(self, envs, subproc_worker, policy, device):
        self.envs = envs
        self.num_processes = len(envs)
        self.policy = policy
        self.device = device
        self.procs = []
        self.parent_pipes = []
        board_size = envs[0].board_size
        self.observation_space = spaces.Box(np.zeros((4, board_size, board_size)), np.ones((4, board_size, board_size)))
        self.action_space = spaces.Discrete(board_size ** 2)

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

    def step(self, *args):
        raise NotImplementedError()

    def test(self, *args):
        raise NotImplementedError()


class PPOTeacherStudentEnvs(Envs):
    def __init__(self, envs, subproc_worker, teacher_policy, student_policy, device):
        super(PPOTeacherStudentEnvs, self).__init__(envs, subproc_worker, teacher_policy, device)
        self.teacher_policy = teacher_policy
        self.student_policy = student_policy
        self.teacher = None
        self.student = None
        self.policy = {}
        self.int_to_str = {-1: 'black', 1: 'white'}

    def reset(self, teacher, win_avg, last_win_avg):
        self.teacher = teacher
        self.student = teacher * -1
        self.policy = {self.teacher: self.teacher_policy, self.student: self.student_policy}
        for pipe in self.parent_pipes:
            pipe.send(('reset', (teacher, win_avg, last_win_avg)))
        for pipe in self.parent_pipes:
            pipe.send(('step', -3))

    def step(self, ex_hiddens_student, ex_hiddens_teacher):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, actions, rewards, dones, infos, outputs = zip(*outs)
        obs, actions, rewards, dones, infos, outputs = list(obs), list(actions), list(rewards), list(dones), list(infos), list(outputs)

        obs = torch.from_numpy(np.array(obs)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).float().to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device).unsqueeze(1)
        outputs = torch.from_numpy(np.array(outputs)).float().to(self.device)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        types = [info['type'] for info in infos]
        while 'need_black_action' in types or 'need_white_action' in types:
            turn = -1 if 'need_black_action' in types else 1
            # print('turn: ', turn)
            # turns = []
            # for type_ in types:
            #     if 'need_black_action' in type_ or 'black_transition' in type_:
            #         turns.append(-1)
            #     if 'need_white_action' in type_ or 'white_transition' in type_:
            #         turns.append(1)
            #     else:
            #         raise ValueError
            choices = [info['choices'] for info in infos]
            ex_hiddens = ex_hiddens_teacher if turn == self.teacher else ex_hiddens_student
            with torch.no_grad():
                v, next_actions, logprob, hidden = self.policy[turn].act(obs, ex_hiddens, masks, choices)
            # next_actions = random_possible_actions(infos)
            for i, info in enumerate(infos):
                if info['type'] != 'need_{}_action'.format(self.int_to_str[turn]):
                    continue
                self.parent_pipes[i].send(('step', int(next_actions[i][0]), (v[i], logprob[i], hidden[i])))
                out = self.parent_pipes[i].recv()
                obs[i] = torch.from_numpy(out[0])
                actions[i] = torch.from_numpy(np.array([out[1]]))
                rewards[i] = torch.from_numpy(np.array([out[2]]))
                dones[i] = out[3]
                infos[i] = out[4]
                outputs[i] = torch.from_numpy(np.array(out[5]))
            types = [info['type'] for info in infos]
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        # while 'need_{}_action'.format(self.int_to_str[turn*-1]) in [info['type'] for info in infos]:
        #     print(infos, dones)
        #     choices = [info['choices'] for info in infos]
        #     ex_hiddens = ex_hiddens_teacher if turn*-1 == self.teacher else ex_hiddens_student
        #     with torch.no_grad():
        #         v, next_actions, logprob, hidden = self.policy[turn*-1].act(obs, ex_hiddens, masks, choices)
        #     # next_actions = random_possible_actions(infos)
        #     for i, info in enumerate(infos):
        #         if info['type'] != 'need_{}_action'.format(self.int_to_str[turn*-1]):
        #             continue
        #         self.parent_pipes[i].send(('step', int(next_actions[i][0]), (v[i], logprob[i], hidden[i])))
        #         out = self.parent_pipes[i].recv()
        #         obs[i] = torch.from_numpy(out[0])
        #         actions[i] = torch.from_numpy(np.array([out[1]]))
        #         rewards[i] = torch.from_numpy(np.array([out[2]]))
        #         dones[i] = out[3]
        #         infos[i] = out[4]
        #         outputs[i] = torch.from_numpy(np.array(out[5]))
        #     masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in dones])
        #     bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        # print(infos, dones)
        for pipe in self.parent_pipes:
            pipe.send(('step', -1))

        if not all(dones):
            for i, d in enumerate(dones):
                if d:
                    self.parent_pipes[i].send(('step', -2))
                    # _ = self.parent_pipes[i].recv()

        t_or_s = []
        for info in infos:
            if 'black_transition' == info['type']:
                turn = -1
            elif 'white_transition' == info['type']:
                turn = 1
            else:
                # raise ValueError('unknown type: {}'.format(info['type']))
                # when over
                turn = 1
            if turn == self.teacher:
                t_or_s.append('teacher')
            else:
                t_or_s.append('student')

        return t_or_s, obs, actions, rewards, dones, infos, outputs, masks, bad_masks

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
                v, next_actions, logprob, hidden = self.student_policy.act(obs, ex_hiddens, masks, choices)
            # next_actions = random_possible_actions(infos)
            for i, info in enumerate(infos):
                self.parent_pipes[i].send(('step', int(next_actions[i][0]), (v[i], logprob[i], hidden[i])))
        for pipe in self.parent_pipes:
            pipe.send(('finish-test', None, None))
        num_wins = [info['wins'] for info in infos]
        print(num_wins)
        return proc_num_games*len(self.parent_pipes), sum(num_wins)


# self-play
class PPOEnvs(Envs):
    def step(self, ex_hiddens):
        outs = [pipe.recv() for pipe in self.parent_pipes]
        obs, actions, rewards, dones, infos, outputs = zip(*outs)
        obs, actions, rewards, dones, infos, outputs = list(obs), list(actions), list(rewards), list(dones), list(infos), list(outputs)

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


class RainbowEnvs(Envs):
    def test(self, name, num_games):
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
            choices = [info['choices'] for info in infos]
            with torch.no_grad():
                next_actions = self.policy.get_test_actions_with_possible_moves(obs, choices)
            # next_actions = random_possible_actions(infos)
            for i, info in enumerate(infos):
                self.parent_pipes[i].send(('step', int(next_actions[i][0]), (None, None, None)))
        for pipe in self.parent_pipes:
            pipe.send(('finish-test', None, None))
        num_wins = [info['wins'] for info in infos]
        print(num_wins)
        return proc_num_games*len(self.parent_pipes), sum(num_wins)