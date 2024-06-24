import copy
import torch
import random
import os

from envs.core_env import BaseEnv
from trainer.logger import Logger
from trainer.replay_memory import ReplayMemory
from trainer.policy_memory import PolicyMemory
from validation.exhaustive_search import NOMA_Searcher
from tqdm import tqdm
from torch.distributions import Categorical


class Trainer:
    def __init__(
        self,
        env: BaseEnv,
        env_bl: BaseEnv,
        model: torch.nn.Module,
        validation_seeds: list,
        metric: str = "MSR",
        accelerator: str = "cpu",
        batch_size: int = 40,
        num_users: int = 10,
        num_channels: int = 15,
        num_epochs: int = 10,
        num_episodes: int = 10000,
        num_tests: int = 10,
        validate_every: int = 200,
        T: int = 3,
        error_threshold: float = 1e-1,
        loss_threshold: float = 1,
        epsilon: float = 1,
        save_dir: str = "simulations",
        save_every: int = 10,
        method: str = "Policy Gradient",
        learning_rate: float = 1e-4,
    ):
        """
        Creates two sets of training-purpose objects: baseline objects and testing objects.
        Each comprises `env` and `model`.
        """
        self.method = method

        self.batch_size = batch_size
        self.N = num_users
        self.K = num_channels

        # Testing objects
        self.env = env
        self.online_model = model  # act randomly based on the given distribution
        self.sync_every = 1e2

        # Baseline objects
        self.target_model = copy.deepcopy(self.online_model)
        self.env_bl = env_bl

        self.metric = metric  # "MSR" or "MMR"
        self.save_dir = save_dir
        self.logger = Logger(save_dir=save_dir, save_every=save_every)
        self.buffer = ReplayMemory(batch_size=self.batch_size)
        self.memory = PolicyMemory(batch_size=self.batch_size)

        self.num_epochs = num_epochs
        self.num_episodes = num_episodes
        self.num_tests = num_tests
        self.validate_every = validate_every
        self.validation_seeds = validation_seeds

        # Exhaustive Search for Validation Environments
        preprocessed_dir = "preprocessed"
        self.preprocessed_dir = preprocessed_dir
        for seed in self.validation_seeds:
            file_name = f"{seed}_{self.N}x{self.K}.txt"
            file_dir = os.path.join(preprocessed_dir, file_name)
            if not os.path.exists(preprocessed_dir):
                os.makedirs(preprocessed_dir)
            if not os.path.isfile(file_dir):
                print(f"Preprocessing: {file_name}")
                searcher = NOMA_Searcher(
                    num_users=self.N, num_channels=self.K, seed=seed
                )
                min, avg, max = searcher.exhaustive_search()

                with open(file_dir, "a") as f:
                    f.write("MIN: " + str(min) + "\n")
                    f.write("AVG: " + str(avg) + "\n")
                    f.write("MAX: " + str(max))

        self.T = T
        self.counts_for_T = 0
        self.error_threshold = error_threshold
        self.loss_threshold = loss_threshold
        self.stopping_criteria = False
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999975

        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.lr)
        self.loss_func = torch.nn.SmoothL1Loss()

        if accelerator not in ["cpu", "mps", "gpu", "cuda"]:
            raise Exception(
                "`accelerator` should be either 'cpu', 'mps', 'cuda', or 'gpu'."
            )
        elif accelerator == "gpu" or accelerator == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise Exception("`cuda` not available")
        elif accelerator == "mps":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                raise Exception("`mps` not available")
        else:  # accelerator == "cpu"
            self.device = torch.device("cpu")

        self.online_model.to(self.device)
        self.target_model.to(self.device)
        self.max_val = 0
        self.min_val = 1000

    def train_test(self):
        sum = []
        episodes = tqdm(range(self.num_episodes))
        for episode in episodes:
            if episode % self.sync_every == 0:
                self.sync_networks()

            state, _ = self.env.reset()
            state = state.unsqueeze(0)

            history = []
            for _ in range(self.N):
                out = self.online_model(state)

                # NOTE: Action (with `Online` network)
                valid_actions_mask = out != float("-inf")
                valid_actions_mask = valid_actions_mask.view(-1)
                valid_indices = torch.nonzero(valid_actions_mask)

                # Exploration
                if torch.rand(1) < self.epsilon:
                    action = random.choice(valid_indices)
                # Exploitation
                else:
                    pred_reward, action = torch.max(out, dim=1)
                self.epsilon = self.epsilon * self.epsilon_decay
                self.epsilon = max(self.epsilon_min, self.epsilon)

                # history.append([prev_state.clone(), state.clone(), torch.tensor([action])])

                # Action
                # (prev_state, state), reward, info, _ = self.env.step(action)
                next_state, reward, info, done = self.env.step(action)
                history.append(
                    [
                        state,
                        next_state,
                        torch.tensor([action]),
                        torch.tensor([done]),
                    ]
                )

                state = next_state.unsqueeze(0)

                # NOTE: Learn
                if self.buffer.get_len() >= 1e2:
                    m_state, m_next_state, m_action, m_done, m_reward = (
                        self.buffer.sample_from_memory()
                    )

                    m_reward = m_reward.squeeze(1).to(self.device)
                    expected_reward = self.td_estimate(m_state, m_action)
                    target_reward = self.td_target(m_reward, m_next_state, m_done)

                    # loss = self.loss_func(expected_reward, m_reward)
                    loss = self.loss_func(expected_reward, target_reward)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            sum_rate = 0
            for i, idx in enumerate(info["usr_idx_history"]):
                usr_info = info["user_info"][idx]
                data_rate = usr_info["data_rate"] / 1e6
                data_rate = torch.tensor([data_rate], dtype=torch.float32)
                sum_rate = sum_rate + data_rate

                # print(idx.item(), usr_info["channel"], data_rate, usr_info["power"], usr_info["distance"], usr_info["CNR"])
                history[i].append(data_rate)
                m_state = history[i][0]
                m_next_state = history[i][1]
                m_action = history[i][2]
                m_done = history[i][3]
                m_reward = history[i][4]
                self.buffer.save_into_memory(
                    m_state, m_next_state, m_action, m_done, m_reward
                )

            if episode % 100 == 0 and self.buffer.get_len() >= 1e2:
                sum.append(sum_rate)
                print(f"EP {episode}: {loss}, {sum_rate}")

    def validate(self):
        counts = 0
        for seed in self.validation_seeds:
            file_name = f"{seed}_{self.N}x{self.K}.txt"
            file_dir = os.path.join(self.preprocessed_dir, file_name)
            with open(file_dir, "r") as f:
                min = float(f.readline().strip("MIN: \n"))
                avg = float(f.readline().strip("AVG: \n"))
                max = float(f.readline().strip("MAX: \n"))

            state, info = self.env.reset(seed)

            reward = 0
            for _ in range(self.N):
                state = state.unsqueeze(0)
                probs = self.target_model(state)
                action = torch.argmax(probs)

                state, reward, info, done = self.env.step(action)

            if reward == 0:
                raise KeyError

            is_match = False
            error_rate = (max - reward) / (max - min)
            if error_rate <= self.error_threshold:
                is_match = True
                counts += 1
            print(f"MAX: {max}\t... {reward}\t {is_match}\t {error_rate:.4f}")

        if counts == len(self.validation_seeds):
            self.counts_for_T += 1
            print("Stopping Criteria Met")
        else:
            self.counts_for_T = 0

    def fit(self):
        if self.metric == "MSR":
            log_name = "sum_rate"
        elif self.metric == "MMR":
            log_name = "min_rate"
        else:
            raise KeyError()

        avg_loss = 0
        avg_reward = 0
        max_reward = 0
        min_reward = 1000

        ep = -1
        while True:
            episodes = tqdm(range(self.validate_every))
            for episode in episodes:
                ep += 1
                state, _ = self.env.reset(ep)
                state_bl, _ = self.env_bl.reset(ep)
                state = state.unsqueeze(0)
                state_bl = state_bl.unsqueeze(0)

                loss = None
                log_probs = 0
                final_reward = 0
                final_reward_bl = 0
                state_list = []
                state_bl_list = []
                action_list = []
                action_bl_list = []
                for _ in range(self.N):
                    state_list.append(state.squeeze(0))
                    state_bl_list.append(state_bl.squeeze(0))
                    action, log_prob, action_bl = self.action_select(state, state_bl)
                    action_list.append(action)
                    action_bl_list.append(action_bl)
                    log_probs = log_probs + log_prob

                    next_state, reward, _, _ = self.env.step(action)
                    next_state_bl, reward_bl, _, _ = self.env_bl.step(action_bl)
                    state = next_state.unsqueeze(0)
                    state_bl = next_state_bl.unsqueeze(0)

                    final_reward = reward
                    final_reward_bl = reward_bl

                avg_reward += final_reward
                max_reward = max(final_reward, final_reward_bl, max_reward)
                min_reward = min(final_reward, final_reward_bl, min_reward)
                self.max_val = max_reward

                # NOTE: Single Policy Gradient Method:
                # if you're up to train with random distance profiles,
                # please make sure to comment out the following lines and go to line 238.
                # loss = 0
                # loss = self.policy_gradient(
                #     log_probs, torch.tensor([final_reward]), torch.tensor([final_reward_bl])
                # )
                # avg_loss += loss

                self.memory.save_into_memory(
                    torch.stack(state_list),
                    torch.stack(action_list),
                    torch.tensor([final_reward], dtype=torch.float32),
                    torch.tensor([final_reward_bl], dtype=torch.float32),
                )

                if final_reward >= final_reward_bl:
                    self.sync_networks()

                # NOTE: Batch Policy Gradient Method
                if self.memory.get_len() >= self.batch_size:
                    loss = self.learn_policy()
                    avg_loss += loss

                if ep % 10 == 0:
                    self.logger.log_step(value=loss, log="loss")
                    self.logger.log_step(value=final_reward, log=log_name)

            avg_loss /= self.validate_every
            avg_reward /= self.validate_every
            print(f"EP: {ep}: {avg_loss}, {avg_reward}, {min_reward} ~ {max_reward}")
            self.logger.log_step(value=avg_loss, log="avg_loss")
            self.logger.log_step(value=avg_reward, log=f"avg_{log_name}")
            tmp = avg_loss
            avg_loss = 0
            avg_reward = 0
            self.validate()

            if self.counts_for_T >= self.T and abs(tmp) <= self.loss_threshold:
                self.stopping_criteria = True

            if self.stopping_criteria:  # while loop break
                break

        self.logger.save()
        torch.save(self.target_model.state_dict(), f"{self.save_dir}/weights")

    def action_select(self, state, state_bl):
        """Policy Gradient (REINFORCE) Method:
        Online model samples from the model's probability distribution
        and baseline model chooses the action corresponding to the highest probability.
        """
        # Online model
        probs = self.online_model(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Baseline model
        probs_bl = self.target_model(state_bl)
        action_bl = torch.argmax(probs_bl)

        return action, log_prob, action_bl

    def policy_gradient(self, log_probs, reward, reward_bl):
        """REINFORCE Algorithm"""
        reward = reward.to(self.device)
        reward_bl = reward_bl.to(self.device)
        loss = (reward_bl - reward) * log_probs
        # loss = (reward - reward_bl) * log_probs
        # loss = -log_probs * reward
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        return loss.item()

    def learn_policy(self):
        log_probs = torch.zeros(self.batch_size).to(self.device)
        history = self.memory.sample_from_memory()
        state, action, reward, reward_bl = history

        for i in range(self.N):
            cur_state = state[:, i, :, :]
            cur_action = action[:, i, :].squeeze()
            log_prob = self.feedforward_and_get_log_prob(cur_state, cur_action)
            log_probs = log_probs + log_prob

        loss = self.policy_gradient(log_probs, reward.squeeze(), reward_bl.squeeze())
        return loss

    def action_selection(self, state):
        action_space = self.online_model(state)

        valid_actions_mask = action_space != float("-inf")
        valid_actions_mask = valid_actions_mask.view(-1)
        valid_indices = torch.nonzero(valid_actions_mask)

        # Exploration
        if torch.rand(1) < self.epsilon:
            action = random.choice(valid_indices)
        # Exploitation
        else:
            pred_reward, action = torch.max(action_space, dim=1)

        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        return action

    def learn(self):
        state, next_state, action, done, reward = self.buffer.sample_from_memory()
        td_estimate = self.td_estimate(state, action)
        td_target = self.td_target(reward, next_state, done)

        loss = self.loss_func(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def one_step_feedforward(self, model, state, is_baseline: bool = False):
        """
        Model feedforwards one step given some state and transits to a new timestep.
        This is to differ baseline model and testing model.

        Args:
            model (BaseModel(nn.Module))
            state
            is_baseline (bool=False): to differ baseline model and testing model.

        Returns:
            chosen_state
            chosen_user_idx
            chosen_channel_idx
        """
        policy = model(state)

        if is_baseline:
            action = torch.argmax(policy, dim=1)
        else:
            action = torch.multinomial(policy, 1).squeeze(1)

        probability = policy[torch.arange(policy.size(0)), action]

        return action, probability

    def feedforward_and_get_log_prob(self, state, action):
        probs = self.online_model(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action)

        return log_prob

    def td_estimate(self, state, action):
        action = action.squeeze()
        return self.online_model(state)[torch.arange(0, self.batch_size), action]

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        reward = reward.to(self.device).squeeze(1)
        done = done.to(self.device).squeeze(1)
        next_state_reward = self.target_model(next_state)
        pred_reward, action = torch.max(next_state_reward, dim=1)
        # next_reward = self.online_model(state, next_state)

        next_reward = self.online_model(next_state)[
            torch.arange(0, self.batch_size), action
        ]
        next_reward = torch.where(done, torch.tensor(0), next_reward)
        val = reward + next_reward
        return val

    def sync_networks(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    def test(self):
        """Run testing process"""
        episodes = tqdm(range(self.num_tests))
        max_reward = 0
        avg_reward = 0
        for episode in episodes:
            state, _ = self.env.reset(2024 + episode)
            state_bl, _ = self.env_bl.reset(2024 + episode)
            state = state.unsqueeze(0)
            state_bl = state_bl.unsqueeze(0)

            final_reward_bl = 0
            for _ in range(self.N):
                action, log_prob, action_bl = self.action_select(state, state_bl)

                next_state, reward, _, _ = self.env.step(action)
                next_state_bl, reward_bl, _, _ = self.env_bl.step(action_bl)
                state = next_state.unsqueeze(0)
                state_bl = next_state_bl.unsqueeze(0)

                final_reward = reward
                final_reward_bl = reward_bl

            max_reward = max(max_reward, final_reward_bl)
            avg_reward += final_reward_bl

            print(f"EP: {episode}: {final_reward_bl}, {max_reward}")
            self.logger.log_step(final_reward_bl, "test", "sum_rate")

        avg_reward /= self.num_tests
        self.logger.save()
        print(f"AVG: {avg_reward}")
