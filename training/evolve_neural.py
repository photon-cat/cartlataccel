"""
Evolve a small neural network controller directly using CMA-ES

Instead of training with PPO, we evolve the weights directly!
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


class SmallNeuralController:
    """
    A small neural network: 3 → 16 → 16 → 1 with tanh activations
    """
    
    def __init__(self, weights):
        """
        weights: flattened array of all network weights
        
        Architecture: 3 → 16 → 16 → 1
        - Layer 1: 3*16 + 16 = 64 params
        - Layer 2: 16*16 + 16 = 272 params  
        - Layer 3: 16*1 + 1 = 17 params
        Total: 353 parameters
        """
        idx = 0
        
        # Layer 1: 3 → 16
        self.W1 = weights[idx:idx+48].reshape(3, 16)
        idx += 48
        self.b1 = weights[idx:idx+16]
        idx += 16
        
        # Layer 2: 16 → 16
        self.W2 = weights[idx:idx+256].reshape(16, 16)
        idx += 256
        self.b2 = weights[idx:idx+16]
        idx += 16
        
        # Layer 3: 16 → 1
        self.W3 = weights[idx:idx+16].reshape(16, 1)
        idx += 16
        self.b3 = weights[idx:idx+1]
    
    @staticmethod
    def n_params():
        return 3*16 + 16 + 16*16 + 16 + 16*1 + 1  # 353
    
    def forward(self, state):
        """Forward pass"""
        # state: [pos, vel, target_pos]
        x = np.tanh(state @ self.W1 + self.b1)  # Layer 1
        x = np.tanh(x @ self.W2 + self.b2)       # Layer 2
        x = np.tanh(x @ self.W3 + self.b3)       # Layer 3 (output)
        return x[0]
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        state: [pos, vel, target_pos]
        We normalize inputs for stability
        """
        # Normalize state
        normalized_state = np.array([
            state[0] / 4.0,      # position (max ~4)
            state[1] / 10.0,     # velocity
            state[2] / 4.0,      # target position
        ])
        
        return self.forward(normalized_state)


def evaluate_neural_controller(weights, n_episodes=5):
    """Evaluate a neural controller"""
    
    controller = SmallNeuralController(weights)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    total_costs = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        actions, targets, actuals = [], [], []
        current_lataccel = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            
            # Calculate target
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            # Get action from neural controller
            action = controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(np.array([action]))
            
            actual_lataccel = info['noisy_action'] if 'noisy_action' in info else action
            
            actions.append(action)
            targets.append(target_lataccel)
            actuals.append(actual_lataccel)
            
            current_lataccel = actual_lataccel
            state = next_state
            
            if terminated or truncated:
                break
        
        # Calculate cost
        costs = calculate_costs(np.array(actuals), np.array(targets), dt=0.02)
        total_costs.append(costs['total_cost'])
    
    return np.mean(total_costs)


def cma_es(objective_fn, n_params, sigma=0.5, max_iterations=100, pop_size=30):
    """CMA-ES for neural network weights"""
    
    # Initialize with small random weights
    mean = np.random.randn(n_params) * 0.1
    C = np.eye(n_params)
    
    best_cost = float('inf')
    best_params = mean.copy()
    
    print(f"Starting CMA-ES with {n_params} parameters, pop_size={pop_size}\n")
    
    for iteration in range(max_iterations):
        # Sample population
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
        
        # Select elite
        n_elite = pop_size // 2
        elite = sorted_population[:n_elite]
        
        # Update mean
        mean = np.mean(elite, axis=0)
        
        # Update covariance
        deviations = np.array([ind - mean for ind in elite])
        C = np.cov(deviations.T)
        if C.ndim == 0:
            C = np.array([[C]])
        
        # Regularize
        C += 1e-6 * np.eye(n_params)
        
        # Progress
        avg_cost = np.mean(sorted_costs[:n_elite])
        print(f"Gen {iteration+1:3d}: Best={best_cost:>8,.0f}, "
              f"Avg(elite)={avg_cost:>8,.0f}, "
              f"Worst={sorted_costs[-1]:>8,.0f}")
        
        # Early stopping if converged
        if iteration > 20 and np.std(sorted_costs[:n_elite]) < 100:
            print(f"\nConverged! Elite std < 100")
            break
    
    return best_params, best_cost


def main():
    print("=" * 80)
    print("EVOLVE NEURAL NETWORK CONTROLLER")
    print("=" * 80)
    print("\nDirect evolution of neural network weights (no gradient descent!)")
    print()
    
    # Get PID baseline
    from controllers import PIDController
    from evolve_controller import evaluate_controller
    
    print("PID Baseline:")
    pid_cost = evaluate_controller([0.195, 0.1, -0.053, 0, 0, 0, 0], n_episodes=10)
    print(f"  {pid_cost:.0f}\n")
    
    # Network info
    n_params = SmallNeuralController.n_params()
    print(f"Network: 3 → 16 → 16 → 1")
    print(f"Parameters: {n_params}")
    print(f"Architecture: Simple feedforward with tanh activations\n")
    
    print("=" * 80)
    print("EVOLUTION")
    print("=" * 80)
    
    start_time = time.time()
    
    best_weights, best_cost = cma_es(
        objective_fn=lambda w: evaluate_neural_controller(w, n_episodes=3),
        n_params=n_params,
        sigma=0.5,
        max_iterations=150,  # More iterations for neural net
        pop_size=30,
    )
    
    elapsed = time.time() - start_time
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    print("\nEvaluating evolved neural controller with 20 episodes...")
    final_cost = evaluate_neural_controller(best_weights, n_episodes=20)
    
    print(f"\nResults:")
    print(f"  PID Baseline:       {pid_cost:>8,.0f}")
    print(f"  Evolved Neural Net: {final_cost:>8,.0f}")
    print(f"  vs PID:             {final_cost/pid_cost:>8.2f}x")
    
    if final_cost < pid_cost:
        improvement = ((pid_cost - final_cost) / pid_cost) * 100
        print(f"  Improvement:        {improvement:>7.1f}%")
    else:
        gap = ((final_cost - pid_cost) / pid_cost) * 100
        print(f"  Gap:               +{gap:>7.1f}%")
    
    print(f"\n  Time: {elapsed:.1f}s")
    
    if final_cost < pid_cost:
        print(f"\n🎉🎉🎉 EVOLVED NEURAL NET BEATS PID! 🎉🎉🎉")
    elif final_cost < pid_cost * 1.1:
        print(f"\n💪 Within 10% of PID!")
    elif final_cost < pid_cost * 1.5:
        print(f"\n✓ Competitive with PID!")
    
    # Save weights
    np.save('evolved_neural_controller.npy', best_weights)
    print(f"\nWeights saved to: evolved_neural_controller.npy")
    
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    print(f"\nPID:                  {pid_cost:>6,.0f} (1.00x)")
    print(f"Evolved Analytical:    ~2,541 (1.18x)")
    print(f"Evolved Neural:       {final_cost:>6,.0f} ({final_cost/pid_cost:.2f}x)")
    print(f"PPO (256x4):          ~2,731 (1.27x)")


if __name__ == "__main__":
    main()

