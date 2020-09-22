import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, num_action):
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, num_action)
        self.fc6 = nn.Linear(128, 1)

        # # actor
        # self.action_layer = nn.Sequential(
        #     nn.Linear(state_dim, n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(n_latent_var, n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(n_latent_var, action_dim),
        #     nn.Softmax(dim=-1)
        # )
        #
        # # critic
        # self.value_layer = nn.Sequential(
        #     nn.Linear(state_dim, n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(n_latent_var, n_latent_var),
        #     nn.Tanh(),
        #     nn.Linear(n_latent_var, 1)
        # )

    def forward(self):
        raise NotImplementedError

    def action_and_value(self, state):
        h = self.conv1(state)
        h = F.relu(h)
        h = self.conv2(h)
        h = F.relu(h)
        h = self.conv3(h)
        h = F.relu(h)
        h = h.view(h.size(0), -1)
        h = self.fc4(h)
        h = F.relu(h)

        pi = self.fc5(h)
        pi = F.softmax(pi, dim=1)

        value = self.fc6(h)
        return pi, value

    # def act(self, state, memory):
    #     state = torch.from_numpy(state).float().to(device)
    #     # action_probs = self.action_layer(state)
    #     action_probs, _ = self.action_and_value(state)
    #     dist = Categorical(action_probs)
    #     action = dist.sample()
    #
    #     memory.states.append(state[0])
    #     memory.actions.append(action)
    #     memory.logprobs.append(dist.log_prob(action))
    #     return action.item()

    def get_action_probs(self, state):
        # state = torch.from_numpy(state).float().to(device)
        action_probs, _ = self.action_and_value(state)
        # dist = Categorical(action_probs)
        # action = dist.sample()
        return action_probs

    def evaluate(self, state, action):
        # action_probs = self.action_layer(state)
        action_probs, state_value = self.action_and_value(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        # state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self,
                 agent_name,
                 board_size,
                 state_channels = 3,
                 lr = 0.001,
                 betas = (0.9, 0.999),
                 gamma = 0.99,  # discount factor
                 K_epochs = 5,  # update policy for K epochs
                 eps_clip = 0.2,  # clip parameter for PPO
                 update_timestep = 2000,  # update policy every n timesteps
                 batch_size = 256,
                 random_seed = None
                 ):
        print('lr:', lr)
        self.env = None

        # self.agent_name = agent_name
        self.agent_name = 'ppo_rand_r-disk_3channels'
        # self.agent_name = 'test'
        self.board_size = board_size
        self.num_action = board_size**2
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep
        self.batch_size = batch_size

        self.policy = ActorCritic(state_channels, self.num_action).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_channels, self.num_action).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        self.memory = Memory()

        self.timestep = 0
        self.episode = 0
        self.avg_loss = 0
        self.total_reward = 0
        self.duration = 0
        self.start = time.time()

        self.writer = SummaryWriter(log_dir="./log/{}".format(self.agent_name))

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(device).detach()

        dataset = torch.utils.data.TensorDataset(rewards, old_states, old_actions, old_logprobs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimize policy for K epochs:
        total_loss = 0
        for _ in range(self.K_epochs):
            for batch_rewards, batch_old_states, batch_old_actions, batch_old_logprobs in dataloader:
                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(batch_old_states, batch_old_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - batch_old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = batch_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, batch_rewards) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.mean()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        avg_loss = total_loss / (self.K_epochs * (self.update_timestep//self.batch_size))
        return avg_loss

    def get_action(self, state):
        possible_moves = self.env.possible_moves
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.policy_old.get_action_probs(state[None])
        possible_action_probs = action_probs[0][possible_moves]
        dist = Categorical(possible_action_probs)
        idx = dist.sample()
        action = torch.tensor(possible_moves[idx])
        logprob = dist.log_prob(idx)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(logprob)
        return action

    def run(self, state, action, reward, done, next_state):
        self.timestep += 1
        self.duration += 1
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)
        self.total_reward += reward
        if self.timestep % self.update_timestep == 0:
            self.avg_loss = self.update()
            self.memory.clear_memory()

        if done:
            self.writer.add_scalar("loss", self.avg_loss, self.episode)
            self.writer.add_scalar("reward", self.total_reward, self.episode)
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.episode)

            elapsed = time.time() - self.start
            text = 'EPISODE: {0:6d} / TOTAL_STEPS: {1:8d} / STEPS: {2:5d} / TOTAL_REWARD: {3:3.0f} / AVG_LOSS: {4:.5f} / STEPS_PER_SECOND: {5:.1f}'.format(
                self.episode + 1, self.timestep, self.duration, self.total_reward, self.avg_loss, self.duration / elapsed)
            self.total_reward = 0
            self.duration = 0
            self.start = time.time()
            self.episode += 1

            print(text)
            with open(self.agent_name + '_output.txt', 'a') as f:
                f.write(text + "\n")




def main():
    ############## Hyperparameters ##############
    env_name = "Breakout-v0"
    # env_name = 'ZaxxonNoFrameskip-v4'
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[-1]
    action_dim = 4
    render = False
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 50000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    # training loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()
        state = state.reshape(state.shape[-1], state.shape[0], state.shape[1])
        for t in range(max_timesteps):
            timestep += 1

            # Running policy_old:
            state = state[None]
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            state = state.reshape(state.shape[-1], state.shape[0], state.shape[1])

            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0

            running_reward += reward
            if render:
                env.render()
            if done:
                break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0


if __name__ == '__main__':
    main()

