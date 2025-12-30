"""
Quick test: Evolve just 2-3 key larger architectures
Faster evolution with fewer generations
"""

import numpy as np
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from eval_cost import calculate_costs
import time

# Register environment
try:
    register(
        id='CartLatAccel-Jerk-v1',
        entry_point='jerk_env:CartLatAccelJerkEnv',
        max_episode_steps=500,
    )
except:
    pass


class NeuralController:
    """Flexible neural network controller"""
    
    def __init__(self, weights, architecture):
        self.architecture = architecture
        self.layers = []
        
        idx = 0
        for i in range(len(architecture) - 1):
            in_size = architecture[i]
            out_size = architecture[i + 1]
            
            W_size = in_size * out_size
            W = weights[idx:idx + W_size].reshape(in_size, out_size)
            idx += W_size
            
            b = weights[idx:idx + out_size]
            idx += out_size
            
            self.layers.append((W, b))
    
    @staticmethod
    def n_params(architecture):
        total = 0
        for i in range(len(architecture) - 1):
            total += architecture[i] * architecture[i + 1] + architecture[i + 1]
        return total
    
    def forward(self, state):
        x = state
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            x = np.tanh(x)
        return x[0] if x.shape == (1,) else x
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        normalized_state = np.array([
            state[0] / 4.0,
            state[1] / 10.0,
            state[2] / 4.0,
        ])
        return self.forward(normalized_state)


def evaluate(weights, architecture, n_episodes=3):
    """Quick evaluation"""
    controller = NeuralController(weights, architecture)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    costs = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        actions, targets, actuals = [], [], []
        current_lataccel = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            action = controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, _, terminated, truncated, info = env.step(np.array([action]))
            actual = info['noisy_action'] if 'noisy_action' in info else action
            
            actions.append(action)
            targets.append(target_lataccel)
            actuals.append(actual)
            current_lataccel = actual
            state = next_state
            
            if terminated or truncated:
                break
        
        cost_dict = calculate_costs(np.array(actuals), np.array(targets), dt=0.02)
        costs.append(cost_dict['total_cost'])
    
    return np.mean(costs)


def quick_evolve(architecture, max_gen=50, pop_size=20):
    """Fast evolution with fewer generations"""
    n_params = NeuralController.n_params(architecture)
    
    mean = np.random.randn(n_params) * 0.1
    sigma = 0.5
    C = np.eye(n_params)
    
    best_cost = float('inf')
    best_params = mean.copy()
    
    print(f"  Evolving {n_params:,} params (pop={pop_size}, gen={max_gen})")
    
    for gen in range(max_gen):
        # Sample
        population = [np.random.multivariate_normal(mean, sigma**2 * C) for _ in range(pop_size)]
        
        # Evaluate
        costs = [evaluate(p, architecture, n_episodes=2) for p in population]
        
        for p, c in zip(population, costs):
            if c < best_cost:
                best_cost = c
                best_params = p.copy()
        
        # Update
        indices = np.argsort(costs)
        n_elite = pop_size // 2
        elite = [population[i] for i in indices[:n_elite]]
        mean = np.mean(elite, axis=0)
        
        deviations = np.array([ind - mean for ind in elite])
        C = np.cov(deviations.T)
        if C.ndim == 0:
            C = np.array([[C]])
        C += 1e-6 * np.eye(n_params)
        
        if (gen + 1) % 10 == 0:
            print(f"    Gen {gen+1}: Best={best_cost:>6,.0f}")
    
    # Final eval
    print(f"  Final eval with 15 episodes...")
    final_cost = evaluate(best_params, architecture, n_episodes=15)
    
    return best_params, final_cost


def main():
    print("=" * 80)
    print("QUICK TEST: Larger Evolved Neural Networks")
    print("=" * 80)
    
    # PID baseline
    from controllers import PIDController
    from evolve_controller import evaluate_controller
    
    print("\nPID Baseline (10 episodes):")
    pid_cost = evaluate_controller([0.195, 0.1, -0.053, 0, 0, 0, 0], n_episodes=10)
    print(f"  {pid_cost:.0f}")
    
    # Test architectures (fewer, more focused)
    architectures = [
        ([3, 32, 32, 1], "32x2"),
        ([3, 64, 64, 1], "64x2"),
        ([3, 32, 32, 32, 1], "32x3"),
    ]
    
    results = []
    
    print("\n" + "=" * 80)
    print("EVOLUTION")
    print("=" * 80)
    
    for arch, name in architectures:
        print(f"\n{name}: {' → '.join(map(str, arch))}")
        
        start = time.time()
        weights, cost = quick_evolve(arch, max_gen=50, pop_size=20)
        elapsed = time.time() - start
        
        results.append({
            'name': name,
            'arch': arch,
            'params': NeuralController.n_params(arch),
            'cost': cost,
            'vs_pid': cost / pid_cost,
            'time': elapsed,
        })
        
        print(f"  Result: {cost:.0f} ({cost/pid_cost:.2f}x vs PID) in {elapsed:.0f}s")
        
        # Save
        filename = f"evolved_{name}.npy"
        np.save(filename, weights)
        print(f"  Saved: {filename}")
        
        if cost / pid_cost < 1.0:
            print(f"\n🎉 BEAT PID!")
            break
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nPID: {pid_cost:.0f}\n")
    print("Network | Params  | Cost   | vs PID  | Time  | Status")
    print("-" * 65)
    
    for r in results:
        marker = "🏆" if r['vs_pid'] < 1.0 else "⭐" if r['vs_pid'] < 1.15 else "✓"
        print(f"{r['name']:<7} | {r['params']:>7,} | {r['cost']:>6,.0f} | {r['vs_pid']:>6.2f}x | {r['time']:>4.0f}s | {marker}")
    
    if results:
        best = min(results, key=lambda x: x['vs_pid'])
        print(f"\nBest: {best['name']} → {best['cost']:.0f} ({best['vs_pid']:.2f}x)")
        
        print("\n" + "=" * 80)
        print("ALL RESULTS")
        print("=" * 80)
        print(f"\nPID:                  {pid_cost:>6,.0f} (1.00x)")
        print(f"Evolved {best['name']}:      {best['cost']:>6,.0f} ({best['vs_pid']:.2f}x) ⭐ NEW")
        print(f"Evolved 16x2:         ~2,457 (1.18x)")
        print(f"Evolved Analytical:   ~2,541 (1.22x)")
        print(f"PPO 256x4:           ~2,731 (1.31x)")


if __name__ == "__main__":
    main()

