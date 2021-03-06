import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.utils import init
from pytorch_a2c_ppo_acktr_gail.a2c_ppo_acktr.distributions import FixedCategorical


thresh = 1e-10


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError
        self.i = 0
        self.e = 0

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, possible_moves, deterministic=False):
        ### action prob in possible moves
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        x = self.dist.linear(actor_features)
        actions = []
        action_log_probs = []
        self.i += 1
        for i, tensor in enumerate(x):
            possible_tensor = tensor[possible_moves[i]]
            dist_ = FixedCategorical(logits=possible_tensor)
            if len(possible_tensor) == 0:
                action = 0
                action_log_prob = torch.tensor(0.0)
                # action_log_prob = torch.tensor(-2.0)
            else:
                if deterministic:
                    idx = dist_.mode()
                else:
                    try:
                        idx = dist_.sample()
                    except:
                        print(possible_tensor, dist_.probs)
                        print(dist_)
                        raise ValueError
                action = possible_moves[i][idx]
                action_log_prob = dist_.log_probs(torch.LongTensor([idx]))
                # if len(possible_tensor) == 1:
                #     action_log_prob = torch.tensor(-2.0)
            actions.append([action])
            action_log_probs.append(action_log_prob)

        action = torch.LongTensor(actions).to(torch.device(value.device))
        action_log_probs = torch.Tensor(action_log_probs).to(torch.device(value.device))
        if self.i % 480 == 0:
            print()
            print('logits, probs')
            print(x[0], dist.probs[0])



        ##### action prob among whole actions
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)
        # x = self.dist.linear(actor_features)
        # actions = []
        # self.i += 1
        # for i, tensor in enumerate(x):
        #     possible_tensor = tensor[possible_moves[i]]
        #     dist_ = FixedCategorical(logits=possible_tensor)
        #     if len(possible_tensor) == 0:
        #         action = 0
        #     else:
        #         if deterministic:
        #             idx = dist_.mode()
        #         else:
        #             try:
        #                 idx = dist_.sample()
        #             except:
        #                 print(possible_tensor, dist_.probs)
        #                 print(dist_)
        #                 raise ValueError
        #         action = possible_moves[i][idx]
        #     actions.append([action])
        #
        # action = torch.LongTensor(actions).to(torch.device(value.device))
        # # print(action, dist.probs)
        # action_log_probs = dist.log_probs(action)
        # with torch.no_grad():
        #     probs = dist.probs[range(len(dist.probs)), action.squeeze()]
        #     probs[probs <= thresh] = 0
        #     probs[probs > thresh] = 1
        # action_log_probs = action_log_probs.squeeze() * probs
        # action_log_probs = action_log_probs.unsqueeze(1)
        # if self.i % 120 == 0:
        #     print('##', x[0], dist.probs[0])


        ## w/o possible moves
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)
        #
        # if deterministic:
        #     action = dist.mode()
        # else:
        #     action = dist.sample()
        #
        # action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, choices):
        ##### action prob in possible moves
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        x = self.dist.linear(actor_features)
        action_log_probs = []
        # self.i += 1
        for i, tensor in enumerate(x):
            possible_tensor = tensor[choices[i]]
            dist_ = FixedCategorical(logits=possible_tensor)
            if len(possible_tensor) == 0 or action[i][0] not in choices[i]:
                action_log_prob = torch.tensor([[0.0]])
            else:
                idx = choices[i].index(action[i][0])
                action_log_prob = dist_.log_probs(torch.LongTensor([idx]))
            action_log_probs.append(action_log_prob)
        action_log_probs = torch.cat(tuple(action_log_probs)).to(torch.device(value.device))
        self.e += 1
        if self.e % 100 == 0:
            print('## probs', dist.probs[0])
            print('## log probs', action_log_probs.squeeze())
        dist_entropy = dist.entropy().mean()

        ##### action prob in whole
        # value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        # dist = self.dist(actor_features)
        #
        # action_log_probs = dist.log_probs(action)
        # with torch.no_grad():
        #     probs = dist.probs[range(len(dist.probs)), action.squeeze()]
        #     probs[probs <= thresh] = 0
        #     probs[probs > thresh] = 1
        # action_log_probs = action_log_probs.squeeze() * probs
        # # action_log_probs[action_log_probs == 0] = -4
        # action_log_probs = action_log_probs.unsqueeze(1)
        # self.e += 1
        # if self.e % 100 == 0:
        #     print('## probs', dist.probs[0])
        #     print('## log probs', action_log_probs.squeeze())
        # dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 2, stride=1, padding=0)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 2, stride=1, padding=0)), nn.ReLU(), Flatten(),
            init_(nn.Linear(256, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
