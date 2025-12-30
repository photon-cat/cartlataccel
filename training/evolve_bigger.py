"""
Evolve larger neural network controllers using CMA-ES

Test multiple network sizes to find optimal architecture
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
    """
    Flexible neural network controller with variable architecture
    """
    
    def __init__(self, weights, architecture):
        """
        weights: flattened array of all network weights
        architecture: list of layer sizes, e.g., [3, 32, 32, 1]
        """
        self.architecture = architecture
        self.layers = []
        
        idx = 0
        for i in range(len(architecture) - 1):
            in_size = architecture[i]
            out_size = architecture[i + 1]
            
            # Weights
            W_size = in_size * out_size
            W = weights[idx:idx + W_size].reshape(in_size, out_size)
            idx += W_size
            
            # Biases
            b = weights[idx:idx + out_size]
            idx += out_size
            
            self.layers.append((W, b))
    
    @staticmethod
    def n_params(architecture):
        """Calculate total parameters for architecture"""
        total = 0
        for i in range(len(architecture) - 1):
            total += architecture[i] * architecture[i + 1]  # Weights
            total += architecture[i + 1]  # Biases
        return total
    
    def forward(self, state):
        """Forward pass with tanh activations"""
        x = state
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:  # No activation on output layer
                x = np.tanh(x)
            else:
                x = np.tanh(x)  # Keep output in [-1, 1]
        return x[0] if x.shape == (1,) else x
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        state: [pos, vel, target_pos]
        """
        # Normalize inputs
        normalized_state = np.array([
            state[0] / 4.0,      # position
            state[1] / 10.0,     # velocity
            state[2] / 4.0,      # target position
        ])
        
        return self.forward(normalized_state)


def evaluate_neural_controller(weights, architecture, n_episodes=5):
    """Evaluate a neural controller"""
    
    controller = NeuralController(weights, architecture)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    total_costs = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        actions, targets, actuals = [], [], []
        current_lataccel = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            action = controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, terminated, truncated, info = env.step(np.array([action]))
            
            actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
            
            actions.append(action)
            targets.append(target_lataccel)
            actuals.append(actual_lataccel)
            
            current_lataccel = actual_lataccel
            state = next_state
            
            if terminated or truncated:
                break
        
        costs = calculate_costs(np.array(actuals), np.array(targets), dt=0.02)
        total_costs.append(costs['total_cost'])
    
    return np.mean(total_costs)


def cma_es(objective_fn, n_params, sigma=0.5, max_iterations=100, pop_size=30):
    """CMA-ES optimization"""
    
    # Initialize
    mean = np.random.randn(n_params) * 0.1
    C = np.eye(n_params)
    
    best_cost = float('inf')
    best_params = mean.copy()
    
    print(f"  Evolving {n_params} parameters with pop_size={pop_size}")
    
    for iteration in range(max_iterations):
        # Sample
        population = []
        for _ in range(pop_size):
            sample = np.random.multivariate_normal(mean, sigma**2 * C)
            population.append(sample)
        
        # Evaluate
        costs = []
        for params in population:
            cost = objective_fn(params)
            costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
        
        # Sort
        indices = np.argsort(costs)
        sorted_population = [population[i] for i in indices]
        sorted_costs = [costs[i] for i in indices]
        
        # Update
        n_elite = pop_size // 2
        elite = sorted_population[:n_elite]
        mean = np.mean(elite, axis=0)
        
        deviations = np.array([ind - mean for ind in elite])
        C = np.cov(deviations.T)
        if C.ndim == 0:
            C = np.array([[C]])
        C += 1e-6 * np.eye(n_params)
        
        # Progress
        if (iteration + 1) % 10 == 0 or iteration == 0:
            avg_cost = np.mean(sorted_costs[:n_elite])
            print(f"    Gen {iteration+1:3d}: Best={best_cost:>7,.0f}, Avg={avg_cost:>7,.0f}")
        
        # Early stopping
        if iteration > 30 and np.std(sorted_costs[:n_elite]) < 100:
            print(f"    Converged at generation {iteration+1}")
            break
    
    return best_params, best_cost


def test_architecture(architecture, name, max_iterations=150):
    """Test a specific architecture"""
    
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"{'='*80}")
    
    n_params = NeuralController.n_params(architecture)
    print(f"Architecture: {' → '.join(map(str, architecture))}")
    print(f"Parameters: {n_params:,}")
    
    start_time = time.time()
    
    # Evolve
    best_weights, training_cost = cma_es(
        objective_fn=lambda w: evaluate_neural_controller(w, architecture, n_episodes=3),
        n_params=n_params,
        sigma=0.5,
        max_iterations=max_iterations,
        pop_size=min(30, 4 + int(3 * np.log(n_params))),  # Scale pop size
    )
    
    # Final evaluation
    print(f"\n  Final evaluation (20 episodes)...")
    final_cost = evaluate_neural_controller(best_weights, architecture, n_episodes=20)
    
    elapsed = time.time() - start_time
    
    print(f"  Training cost: {training_cost:>7,.0f}")
    print(f"  Final cost:    {final_cost:>7,.0f}")
    print(f"  Time:          {elapsed:>7.1f}s")
    
    # Save weights
    filename = f"evolved_{name.lower().replace(' ', '_')}.npy"
    np.save(filename, best_weights)
    print(f"  Saved: {filename}")
    
    return {
        'name': name,
        'architecture': architecture,
        'n_params': n_params,
        'cost': final_cost,
        'time': elapsed,
        'weights_file': filename,
    }


def main():
    print("=" * 80)
    print("EVOLVE LARGER NEURAL NETWORKS")
    print("=" * 80)
    print("\nTesting multiple architectures to find optimal size")
    
    # Get PID baseline
    from controllers import PIDController
    from evolve_controller import evaluate_controller
    
    print("\nPID Baseline:")
    pid_cost = evaluate_controller([0.195, 0.1, -0.053, 0, 0, 0, 0], n_episodes=20)
    print(f"  {pid_cost:.0f}")
    
    # Test different architectures
    architectures = [
        ([3, 16, 16, 1], "Small (16x2)"),
        ([3, 32, 32, 1], "Medium (32x2)"),
        ([3, 64, 64, 1], "Large (64x2)"),
        ([3, 32, 32, 32, 1], "Deep (32x3)"),
        ([3, 64, 32, 1], "Wide-Narrow (64→32)"),
        ([3, 128, 1], "Single Wide (128)"),
    ]
    
    results = []
    
    print("\n" + "=" * 80)
    print("EVOLUTION EXPERIMENTS")
    print("=" * 80)
    
    for arch, name in architectures:
        try:
            result = test_architecture(arch, name, max_iterations=150)
            result['vs_pid'] = result['cost'] / pid_cost
            results.append(result)
            
            print(f"\n  → Result: {result['cost']:.0f} ({result['vs_pid']:.2f}x vs PID)")
            
            # Check if we beat PID
            if result['vs_pid'] < 1.0:
                print(f"\n🎉🎉🎉 WE BEAT PID WITH {name}! 🎉🎉🎉")
                break
                
        except KeyboardInterrupt:
            print("\n\nStopped by user")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nPID Baseline: {pid_cost:.0f}\n")
    print("Architecture         | Params  | Cost   | vs PID  | Time   | Winner")
    print("-" * 80)
    
    for r in results:
        arch_str = ' → '.join(map(str, r['architecture']))
        marker = "🏆" if r['vs_pid'] < 1.0 else "⭐" if r['vs_pid'] < 1.15 else "✓" if r['vs_pid'] < 1.3 else ""
        print(f"{arch_str:<20} | {r['n_params']:>7,} | {r['cost']:>6,.0f} | "
              f"{r['vs_pid']:>6.2f}x | {r['time']:>5.0f}s | {marker}")
    
    if results:
        best = min(results, key=lambda x: x['vs_pid'])
        print(f"\n🎯 BEST: {best['name']}")
        print(f"   Architecture: {' → '.join(map(str, best['architecture']))}")
        print(f"   Cost: {best['cost']:.0f} ({best['vs_pid']:.2f}x vs PID)")
        print(f"   Weights: {best['weights_file']}")
        
        if best['vs_pid'] < 1.0:
            improvement = ((pid_cost - best['cost']) / pid_cost) * 100
            print(f"\n🎉 BEATS PID by {improvement:.1f}%!")
        elif best['vs_pid'] < 1.1:
            gap = best['cost'] - pid_cost
            print(f"\n💪 Very close! Only {gap:.0f} cost away from PID")
        
        # Compare with previous results
        print("\n" + "=" * 80)
        print("COMPARISON WITH OTHER METHODS")
        print("=" * 80)
        print(f"\nPID:                    {pid_cost:>6,.0f} (1.00x)")
        print(f"Best Evolved Neural:    {best['cost']:>6,.0f} ({best['vs_pid']:.2f}x) ⭐ NEW")
        print(f"Previous Evolved (16x2): ~2,457 (1.18x)")
        print(f"Evolved Analytical:     ~2,541 (1.22x)")
        print(f"PPO (256x4):           ~2,731 (1.31x)")
        
        if best['vs_pid'] < 1.18:
            print(f"\n✨ NEW BEST! Improved from 1.18x to {best['vs_pid']:.2f}x")


if __name__ == "__main__":
    main()

