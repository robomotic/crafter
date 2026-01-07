import datetime
import json
import pathlib

import imageio
import numpy as np

try:
  import gymnasium as gym
  BaseWrapper = gym.Wrapper
except ImportError:
  BaseWrapper = object


def _unwrap_env(env):
  while hasattr(env, '_env'):
    env = env._env
  return env


class Recorder(BaseWrapper):

  def __init__(
      self, env, directory, save_stats=True, save_video=True,
      save_episode=True, video_size=(512, 512), gymnasium_api=False):
    self._gymnasium_api = bool(gymnasium_api)
    if directory and save_stats:
      env = StatsRecorder(env, directory, gymnasium_api=self._gymnasium_api)
    if directory and save_video:
      env = VideoRecorder(env, directory, video_size, gymnasium_api=self._gymnasium_api)
    if directory and save_episode:
      env = EpisodeRecorder(env, directory, gymnasium_api=self._gymnasium_api)
    if BaseWrapper is not object:
      super().__init__(env)
    self._env = env

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self, **kwargs):
    result = self._env.reset(**kwargs)
    obs, info = result if isinstance(result, tuple) else (result, {})
    if self._gymnasium_api or kwargs:
      return obs, info
    return obs

  def step(self, action):
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, done, info = result
      terminated, truncated = done, False
    done = terminated or truncated
    if self._gymnasium_api:
      return obs, reward, terminated, truncated, info
    return obs, reward, done, info


class StatsRecorder(BaseWrapper):

  def __init__(self, env, directory, gymnasium_api=False):
    self._gymnasium_api = bool(gymnasium_api)
    if BaseWrapper is not object:
      super().__init__(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._file = (self._directory / 'stats.jsonl').open('a')
    self._length = None
    self._reward = None
    self._unlocked = None
    self._stats = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self, **kwargs):
    result = self._env.reset(**kwargs)
    obs, info = result if isinstance(result, tuple) else (result, {})
    self._length = 0
    self._reward = 0
    self._unlocked = None
    self._stats = None
    if self._gymnasium_api or kwargs:
      return obs, info
    return obs

  def step(self, action):
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, done, info = result
      terminated, truncated = done, False
    done = terminated or truncated
    self._length += 1
    self._reward += info['reward']
    if done:
      self._stats = {'length': self._length, 'reward': round(self._reward, 1)}
      for key, value in info['achievements'].items():
        self._stats[f'achievement_{key}'] = value
      self._save()
    if self._gymnasium_api:
      return obs, reward, terminated, truncated, info
    return obs, reward, done, info

  def _save(self):
    self._file.write(json.dumps(self._stats) + '\n')
    self._file.flush()


class VideoRecorder(BaseWrapper):

  def __init__(self, env, directory, size=(512, 512), gymnasium_api=False):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env, gymnasium_api=gymnasium_api)
    self._gymnasium_api = bool(gymnasium_api)
    if BaseWrapper is not object:
      super().__init__(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._size = size
    self._frames = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self, **kwargs):
    result = self._env.reset(**kwargs)
    obs, info = result if isinstance(result, tuple) else (result, {})
    base_env = _unwrap_env(self._env)
    self._frames = [base_env.render(self._size)]
    if self._gymnasium_api or kwargs:
      return obs, info
    return obs

  def step(self, action):
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, done, info = result
      terminated, truncated = done, False
    done = terminated or truncated
    base_env = _unwrap_env(self._env)
    self._frames.append(base_env.render(self._size))
    if done:
      self._save()
    if self._gymnasium_api:
      return obs, reward, terminated, truncated, info
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.mp4'))
    try:
      imageio.mimsave(filename, self._frames)
    except Exception:
      # If no suitable imageio backend is available (e.g. ffmpeg),
      # skip saving video instead of crashing tests or examples.
      try:
        # Fallback: write individual PNG frames if possible.
        for i, frame in enumerate(self._frames):
          p = self._directory / (self._env.episode_name + f'-{i:04d}.png')
          imageio.imwrite(str(p), frame)
      except Exception:
        pass


class EpisodeRecorder(BaseWrapper):

  def __init__(self, env, directory, gymnasium_api=False):
    if not hasattr(env, 'episode_name'):
      env = EpisodeName(env, gymnasium_api=gymnasium_api)
    self._gymnasium_api = bool(gymnasium_api)
    if BaseWrapper is not object:
      super().__init__(env)
    self._env = env
    self._directory = pathlib.Path(directory).expanduser()
    self._directory.mkdir(exist_ok=True, parents=True)
    self._episode = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self, **kwargs):
    result = self._env.reset(**kwargs)
    obs, info = result if isinstance(result, tuple) else (result, {})
    self._episode = [{'image': obs}]
    if self._gymnasium_api or kwargs:
      return obs, info
    return obs

  def step(self, action):
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, done, info = result
      terminated, truncated = done, False
    done = terminated or truncated
    transition = {
        'action': action, 'image': obs, 'reward': reward, 'done': done,
    }
    for key, value in info.items():
      if key in ('inventory', 'achievements'):
        continue
      transition[key] = value
    for key, value in info['achievements'].items():
      transition[f'achievement_{key}'] = value
    for key, value in info['inventory'].items():
      transition[f'inventory_{key}'] = value
    self._episode.append(transition)
    if done:
      self._save()
    if self._gymnasium_api:
      return obs, reward, terminated, truncated, info
    return obs, reward, done, info

  def _save(self):
    filename = str(self._directory / (self._env.episode_name + '.npz'))
    for key, value in self._episode[1].items():
      if key not in self._episode[0]:
        self._episode[0][key] = np.zeros_like(value)
    episode = {k: np.array([step[k] for step in self._episode]) for k in self._episode[0]}
    np.savez_compressed(filename, **episode)


class EpisodeName(BaseWrapper):

  def __init__(self, env, gymnasium_api=False):
    self._gymnasium_api = bool(gymnasium_api)
    if BaseWrapper is not object:
      super().__init__(env)
    self._env = env
    self._timestamp = None
    self._unlocked = None
    self._length = None

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    return getattr(self._env, name)

  def reset(self, **kwargs):
    result = self._env.reset(**kwargs)
    obs, info = result if isinstance(result, tuple) else (result, {})
    self._timestamp = None
    self._unlocked = None
    self._length = 0
    if self._gymnasium_api or kwargs:
      return obs, info
    return obs

  def step(self, action):
    result = self._env.step(action)
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, done, info = result
      terminated, truncated = done, False
    done = terminated or truncated
    self._length += 1
    if done:
      self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
      self._unlocked = sum(int(v >= 1) for v in info['achievements'].values())
    if self._gymnasium_api:
      return obs, reward, terminated, truncated, info
    return obs, reward, done, info

  @property
  def episode_name(self):
    return f'{self._timestamp}-ach{self._unlocked}-len{self._length}'
