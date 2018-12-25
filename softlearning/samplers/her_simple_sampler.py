import numpy as np
import math

from .sampler_base import BaseSampler


class HerSimpleSampler(BaseSampler):
    def __init__(self, replay_k=99, **kwargs):
        super(HerSimpleSampler, self).__init__(**kwargs)

        """ Sampler that can be used for HER experience replay.
        Args:
            replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
                as many HER replays as regular replays are used)
            reward_fun (function): function to re-compute the reward with substituted goals
        """

        self._future_p = 1 - (1. / (1 + replay_k))

        def reward_fun(ag_2, g, info):  # vectorized
            return self.env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

        self._reward_fun = reward_fun

        self._path_length = 0
        self._path_return = 0
        self._infos = []
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0

    def sample(self):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        action = self.policy.actions_np([
            self.env.convert_to_active_observation(
                self._current_observation)[None]
        ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._infos.append(info)
        self._total_samples += 1

        self.pool.add_sample(
            observations=self._current_observation,
            actions=action,
            rewards=reward,
            terminals=terminal,
            next_observations=next_observation)

        if terminal or self._path_length >= self._max_path_length:
            last_path = self.pool.last_n_batch(
                self._path_length,
                observation_keys=getattr(self.env, 'observation_keys', None))
            last_path.update({'infos': self._infos})

            self.add_her_samples(last_path, self._path_length)


            self._last_n_paths.appendleft(last_path)

            self.policy.reset()
            self._current_observation = self.env.reset()

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self._path_length = 0
            self._path_return = 0
            self._infos = []

            self._n_episodes += 1

        else:
            self._current_observation = next_observation

        return self._current_observation, reward, terminal, info


    def add_her_samples(self, path, path_length):
        print(self._future_p)
        print(self._reward_fun)
        print(path_length)
        her_sample_size = math.ceil(self._future_p*path_length)
        t_samples =  np.random.randint(path_length, size=her_sample_size)

        for key in path.keys():
            print(path[key][t_samples])


        transitions = {key: path[key][t_samples].copy() for key in path.keys()}

        print(transitions)





    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        random_batch = self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

        return random_batch

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
