import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import gym_cartlataccel
from model import ActorCritic
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from tensordict import TensorDict

def select_device(device_pref: str):
  """Pick a device, falling back to CPU if the GPU arch is unsupported by the installed torch build."""
  def try_cuda():
    try:
      if not torch.cuda.is_available():
        return None, "cuda unavailable (is_available returned False)"
      major, minor = torch.cuda.get_device_capability()
      arch = f"sm_{major}{minor}"
      supported_arches = set(torch.cuda.get_arch_list())
      if arch in supported_arches:
        return torch.device("cuda"), f"cuda ({arch})"
      return None, f"CUDA arch {arch} not supported by this torch build ({sorted(supported_arches)}), falling back to CPU."
    except Exception as e:
      return None, f"cuda unavailable ({e}), falling back to CPU."

  if device_pref == "cpu":
    return torch.device("cpu"), "cpu (requested)"
  if device_pref == "cuda":
    dev, msg = try_cuda()
    return (dev or torch.device("cpu")), msg if dev else msg

  # auto
  dev, msg = try_cuda()
  if dev:
    return dev, msg
  print(msg)
  return torch.device("cpu"), "cpu (fallback)"

class PPO:
  def __init__(self, env, model, lr=1e-1, gamma=0.99, lam=0.95, clip_range=0.2, epochs=1, n_steps=30, ent_coeff=0.01, bs=30, env_bs=1, device='cpu', debug=False):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    storage_dir = os.path.join(os.path.dirname(__file__), "rb_storage")
    os.makedirs(storage_dir, exist_ok=True)
    os.environ.setdefault("TMPDIR", storage_dir)
    self.replay_buffer = None
    self.buffer_disabled_reason = None
    try:
      self.replay_buffer = ReplayBuffer(
        storage=LazyMemmapStorage(max_size=10000, scratch_dir=storage_dir),
        batch_size=bs
      )
    except Exception as e:
      self.buffer_disabled_reason = str(e)
      print(f"ReplayBuffer disabled ({e}); falling back to direct updates.")
    self.bs = bs
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug

  def compute_gae(self, rewards, values, done, next_value):
    returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma*next_value*(1-done[t]) - values[t]
      gae = delta + self.gamma*self.lam*(1-done[t])*gae
      advantages[t] = gae
      returns[t] = gae + values[t]
      next_value = values[t]
    return returns, advantages

  def evaluate_cost(self, states, actions, returns, advantages, logprob):
    new_logprob = self.model.actor.get_logprob(states, actions)
    entropy = (torch.log(self.model.actor.std) + 0.5 * (1 + torch.log(torch.tensor(2 * torch.pi)))).sum(dim=-1)
    ratio = torch.exp(new_logprob-logprob).squeeze()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(self.model.critic(states).squeeze(), returns)
    entropy_loss = -self.ent_coeff * entropy.mean()
    return {"actor": actor_loss, "critic": critic_loss, "entropy": entropy_loss}

  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device=None):
    if device is None:
      device = next(model.parameters()).device
    states, actions, rewards, dones = [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      state_tensor = torch.FloatTensor(state).to(device)
      action = model.get_action(state_tensor, deterministic=deterministic)
      next_state, reward, terminated, truncated, info = env.step(action)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      done = terminated or truncated
      dones.append(done)

      state = next_state
      if done:
        state, _ = env.reset()
    return states, actions, rewards, dones, next_state

  def train(self, max_evals=1000):
    eps = 0
    while True:
      # rollout
      start = time.perf_counter()
      states, actions, rewards, dones, next_state = self.rollout(self.env, self.model.actor, self.n_steps, device=self.device)
      rollout_time = time.perf_counter()-start

      # compute gae
      start = time.perf_counter()
      with torch.no_grad():
        states_arr = np.asarray(states)  # (n_steps, env_bs, obs_dim)
        actions_arr = np.asarray(actions)  # (n_steps, env_bs, act_dim)
        rewards_arr = np.asarray(rewards)  # (n_steps, env_bs)
        state_tensor_flat = torch.FloatTensor(states_arr.reshape(-1, states_arr.shape[-1])).to(self.device)
        action_tensor_flat = torch.FloatTensor(actions_arr.reshape(-1, actions_arr.shape[-1])).to(self.device)

        values = self.model.critic(state_tensor_flat).cpu().numpy().reshape(self.n_steps, self.env_bs)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_values = self.model.critic(next_state_tensor).cpu().numpy().squeeze()  # (env_bs,)

        self.model.actor.std = self.model.actor.log_std.exp().to(self.device) # update std
        logprobs = self.model.actor.get_logprob(state_tensor_flat, action_tensor_flat).cpu().numpy().reshape(self.n_steps, self.env_bs)

      # compute gae per env
      returns = np.zeros_like(rewards_arr)
      advantages = np.zeros_like(rewards_arr)
      for env_idx in range(self.env_bs):
        r_env = rewards_arr[:, env_idx]
        v_env = values[:, env_idx]
        nv = next_values[env_idx] if np.ndim(next_values) else next_values
        ret_env, adv_env = self.compute_gae(r_env, v_env, np.array(dones), nv)
        returns[:, env_idx] = ret_env
        advantages[:, env_idx] = adv_env

      # flatten back to match tensors
      returns_flat = torch.FloatTensor(returns.reshape(-1)).to(self.device)
      advantages_flat = torch.FloatTensor(advantages.reshape(-1)).to(self.device)
      advantages_flat = (advantages_flat - torch.mean(advantages_flat)) / (torch.std(advantages_flat) + 1e-8)
      logprobs_flat = torch.FloatTensor(logprobs.reshape(-1)).to(self.device)
      gae_time = time.perf_counter()-start

      if self.replay_buffer:
        start = time.perf_counter()
        episode_dict = TensorDict(
          {
            "states": state_tensor_flat,
            "actions": action_tensor_flat,
            "returns": returns_flat,
            "advantages": advantages_flat,
            "logprobs": logprobs_flat,
          },
          batch_size=returns_flat.shape[0],
          device=self.device,
        )
        self.replay_buffer.extend(episode_dict)
        buffer_time = time.perf_counter() - start
      else:
        buffer_time = 0.0

      # update
      start = time.perf_counter()
      for _ in range(self.epochs):
        if self.replay_buffer:
          for batch in self.replay_buffer:
            adv = batch['advantages']
            adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)
            costs = self.evaluate_cost(batch['states'], batch['actions'], batch['returns'], adv, batch['logprobs'])
            loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            break
        else:
          costs = self.evaluate_cost(state_tensor_flat, action_tensor_flat, returns_flat, advantages_flat, logprobs_flat)
          loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"]
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
      if self.replay_buffer:
        self.replay_buffer.empty() # clear buffer
      update_time = time.perf_counter() - start

      # debug info
      if self.debug:
        print(f"critic loss {costs['critic'].item():.3f} entropy {costs['entropy'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")

      eps += self.env_bs
      avg_reward = np.sum(rewards)/self.env_bs

      if eps > max_evals:
        print(f"Total time: {time.time() - self.start}")
        break
      else:
        print(f"eps {eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")
        self.hist.append((eps, avg_reward))

    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=30000)
  parser.add_argument("--env_bs", type=int, default=1000)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--render", type=str, default="human")
  parser.add_argument("--debug", default=False)
  parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
  args = parser.parse_args()

  print(f"training ppo with max_evals {args.max_evals}") 
  start = time.time()
  device, device_msg = select_device(args.device)
  print(f"using device: {device_msg}")
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=args.env_bs)
  model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1])
  ppo = PPO(env, model, env_bs=args.env_bs, debug=args.debug, device=device)
  best_model, hist = ppo.train(args.max_evals)
  train_time = time.time() - start

  print(f"rolling out best model") 
  start = time.time()
  env = gym.make("CartLatAccel-v1", noise_mode=args.noise_mode, env_bs=1, render_mode=args.render)
  states, actions, rewards, dones, next_state = ppo.rollout(env, best_model, max_steps=200, deterministic=True, device=device)
  rollout_time = time.time() - start
  print(f"reward {sum(rewards)}")
  print(f"mean action {np.mean(abs(np.array(actions)))}")
  print(f"train time {train_time}, rollout {rollout_time}")

  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
