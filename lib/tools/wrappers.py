import gym
from gym.wrappers import TimeLimit
import numpy as np


class RepeatActionWrapper(gym.Wrapper):
    import pybullet_envs
    def __init__(self, env, action_repeat=3, gamma = 0.99):
        super(RepeatActionWrapper, self).__init__(env)
        self.action_repeat = action_repeat
        self.env_max_steps = env.spec.max_episode_steps
        self.gamma = gamma

    def step(self, action):
        r = 0
        for i in range(self.action_repeat):
            obs_, reward_, done_, info_ = self.env.step(action)
            r = self.gamma * r + reward_
            if done_:
                break; 
        if 'success' not in info_.keys():
            info_['success'] = done_ 
        return obs_, r, done_, info_
    
class FailedTimeOutWrapper(gym.Wrapper):
    import pybullet_envs
    def __init__(self, env):
        super(FailedTimeOutWrapper, self).__init__(env)
        self.env_max_steps = env.spec.max_episode_steps
        self._current_step = 0

    def step(self, action):
        obs_, reward_, done_, info_ = self.env.step(action)
        self._current_step += 1
        if done_ :
            info_['success'] = self._current_step < self.env_max_steps
            self._current_step = 0
            
        return obs_, reward_, done_, info_
    
class DonewithFellenDown(gym.Wrapper):
    import pybullet_envs
    def __init__(self, env):
        super(DonewithFellenDown, self).__init__(env)

    def step(self, action):
        obs_, reward_, done_, info_ = self.env.step(action)
        done_ = done_ or self.env.env._alive < 0
        if done_ :
            info_['success'] = self.env.env._alive >= 0
            
        return obs_, reward_, done_, info_
    
    

class FailedZeroingWrapper(gym.Wrapper):
    import pybullet_envs
    def __init__(self, env):
        super(FailedZeroingWrapper, self).__init__(env)
        self.env_max_steps = env.spec.max_episode_steps
        self._current_step = 0

    def step(self, action):
        obs_, reward_, done_, info_ = self.env.step(action)
        self._current_step += 1
        fails = self._current_step < self.env_max_steps
        if done_:
            info_['timeout'] = done_ and not fails #timeout
            self._current_step = 0
            if fails:
                reward_ = 0
        return obs_, reward_, done_, info_



class DoneOnSuccessWrapper(gym.Wrapper):
    """
    Reset on success and offsets the reward.
    Useful for GoalEnv.
    """
    def __init__(self, env, reward_offset=1.0):
        super(DoneOnSuccessWrapper, self).__init__(env)
        self.reward_offset = reward_offset

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = done or info.get('is_success', False)
        reward += self.reward_offset
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.compute_reward(achieved_goal, desired_goal, info)
        return reward + self.reward_offset


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.

    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.

        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))
