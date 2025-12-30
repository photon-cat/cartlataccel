"""
Evolve a custom controller using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

This optimizes controller parameters directly on the cost function!
"""

import numpy as np
import gymnasium as gym
from jerk_env import CartLatAccelJerkEnv
from gymnasium.envs.registration import register
from eval_cost import calculate_costs
from controllers import PIDController
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


class EvolvableController:
    """
    A more sophisticated controller that can be evolved
    
    Architecture:
    u(t) = P*e + I*∫e + D*de/dt + FF*target + NL*e^3 + V*velocity
    
    Where:
    - e = error (target - actual)
    - FF = feedforward on target
    - NL = nonlinear term (cubic error)
    - V = velocity damping
    """
    
    def __init__(self, params):
        """
        params: [P, I, D, FF, NL, V, bias]
        """
        self.p = params[0]
        self.i = params[1]
        self.d = params[2]
        self.ff = params[3]       # feedforward
        self.nl = params[4]       # nonlinear (cubic)
        self.v = params[5]        # velocity damping
        self.bias = params[6]     # constant bias
        
        self.error_integral = 0
        self.prev_error = 0
        self.reset()
    
    def reset(self):
        self.error_integral = 0
        self.prev_error = 0
    
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        state: [pos, vel, target_pos]
        """
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        
        # PID terms
        pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
        
        # Additional terms
        feedforward = self.ff * target_lataccel
        nonlinear = self.nl * (error ** 3)  # Cubic term for large errors
        velocity_damping = self.v * state[1]  # Damping based on velocity
        
        output = pid_output + feedforward + nonlinear + velocity_damping + self.bias
        
        return output


def evaluate_controller(params, n_episodes=5, noise_mode='medium'):
    """Evaluate a controller on the task"""
    
    controller = EvolvableController(params)
    env = gym.make("CartLatAccel-Jerk-v1", env_bs=1, noise_mode=noise_mode,
                   jerk_penalty_weight=0.005, action_penalty_weight=0.0005)
    
    total_costs = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        controller.reset()
        
        actions, targets, actuals = [], [], []
        current_lataccel = 0.0
        
        for step in range(500):
            pos, vel, target_pos = state[0], state[1], state[2]
            
            # Calculate target lateral acceleration
            target_lataccel = (target_pos - pos) * 10.0
            target_lataccel = np.clip(target_lataccel, -1.0, 1.0)
            
            # Get action from controller
            action = controller.update(target_lataccel, current_lataccel, state, None)
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
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


def cma_es_simple(objective_fn, initial_params, sigma=0.5, max_iterations=100, pop_size=20):
    """
    Simple CMA-ES implementation
    
    Args:
        objective_fn: function to minimize
        initial_params: starting parameters
        sigma: initial step size
        max_iterations: max generations
        pop_size: population size
    """
    
    n_params = len(initial_params)
    
    # Initialize
    mean = np.array(initial_params)
    C = np.eye(n_params)  # Covariance matrix
    
    best_cost = float('inf')
    best_params = mean.copy()
    
    print(f"Starting CMA-ES with {n_params} parameters, pop_size={pop_size}\n")
    
    for iteration in range(max_iterations):
        # Sample population
        population = []
        for _ in range(pop_size):
            sample = np.random.multivariate_normal(mean, sigma**2 * C)
            population.append(sample)
        
        # Evaluate population
        costs = []
        for i, params in enumerate(population):
            cost = objective_fn(params)
            costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_params = params.copy()
        
        # Sort by cost
        indices = np.argsort(costs)
        sorted_population = [population[i] for i in indices]
        sorted_costs = [costs[i] for i in indices]
        
        # Select elite (top 50%)
        n_elite = pop_size // 2
        elite = sorted_population[:n_elite]
        
        # Update mean
        mean = np.mean(elite, axis=0)
        
        # Update covariance (simplified)
        deviations = np.array([ind - mean for ind in elite])
        C = np.cov(deviations.T)
        if C.ndim == 0:  # Handle 1D case
            C = np.array([[C]])
        
        # Regularize to prevent singular matrix
        C += 1e-6 * np.eye(n_params)
        
        # Adapt sigma (simple version)
        # sigma *= 0.99
        
        # Print progress
        avg_cost = np.mean(sorted_costs[:n_elite])
        print(f"Gen {iteration+1:3d}: Best={best_cost:>8,.0f}, "
              f"Avg(elite)={avg_cost:>8,.0f}, "
              f"Worst={sorted_costs[-1]:>8,.0f}, "
              f"sigma={sigma:.3f}")
        
        # Print best params periodically
        if (iteration + 1) % 20 == 0:
            print(f"  Best params: {best_params}")
    
    return best_params, best_cost


def main():
    print("=" * 80)
    print("EVOLUTIONARY CONTROLLER OPTIMIZATION")
    print("=" * 80)
    print("\nUsing CMA-ES to evolve a sophisticated controller!")
    print()
    
    # Get PID baseline
    print("Evaluating PID baseline...")
    pid = PIDController(p=0.195, i=0.1, d=-0.053)
    pid_cost = evaluate_controller([0.195, 0.1, -0.053, 0, 0, 0, 0], n_episodes=10)
    print(f"PID Baseline: {pid_cost:.0f}\n")
    
    # Initial parameters: Start from PID + small random values for extra terms
    initial_params = [
        0.195,   # P
        0.100,   # I
        -0.053,  # D
        0.0,     # FF (feedforward)
        0.0,     # NL (nonlinear)
        0.0,     # V (velocity damping)
        0.0,     # bias
    ]
    
    print("=" * 80)
    print("EVOLUTION")
    print("=" * 80)
    print(f"Initial params: {initial_params}")
    print(f"Optimizing 7 parameters: [P, I, D, FF, NL, V, bias]\n")
    
    # Run evolution
    start_time = time.time()
    
    best_params, best_cost = cma_es_simple(
        objective_fn=lambda p: evaluate_controller(p, n_episodes=3),  # 3 episodes per eval for speed
        initial_params=initial_params,
        sigma=0.3,  # Step size
        max_iterations=50,  # Generations
        pop_size=20,  # Population size
    )
    
    elapsed = time.time() - start_time
    
    # Final evaluation with more episodes
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    
    print("\nEvaluating best controller with 20 episodes...")
    final_cost = evaluate_controller(best_params, n_episodes=20)
    
    print(f"\nResults:")
    print(f"  PID Baseline:      {pid_cost:>8,.0f}")
    print(f"  Evolved Controller: {final_cost:>8,.0f}")
    print(f"  vs PID:            {final_cost/pid_cost:>8.2f}x")
    print(f"  Improvement:       {((pid_cost - final_cost)/pid_cost)*100:>7.1f}%")
    print(f"\n  Time: {elapsed:.1f}s")
    
    print(f"\nBest parameters:")
    print(f"  P  (proportional):  {best_params[0]:>8.4f}")
    print(f"  I  (integral):      {best_params[1]:>8.4f}")
    print(f"  D  (derivative):    {best_params[2]:>8.4f}")
    print(f"  FF (feedforward):   {best_params[3]:>8.4f}")
    print(f"  NL (nonlinear):     {best_params[4]:>8.4f}")
    print(f"  V  (vel damping):   {best_params[5]:>8.4f}")
    print(f"  B  (bias):          {best_params[6]:>8.4f}")
    
    if final_cost < pid_cost:
        print(f"\n🎉🎉🎉 EVOLVED CONTROLLER BEATS PID! 🎉🎉🎉")
        improvement = ((pid_cost - final_cost) / pid_cost) * 100
        print(f"Improvement: {improvement:.1f}%")
    elif final_cost < pid_cost * 1.1:
        print(f"\n💪 Very close to PID! Within 10%")
    else:
        print(f"\n✓ Evolution complete, but PID still wins")
    
    print("\n" + "=" * 80)
    print("USAGE")
    print("=" * 80)
    print("\nTo use the evolved controller:")
    print(f"""
controller = EvolvableController([
    {best_params[0]:.4f},  # P
    {best_params[1]:.4f},  # I
    {best_params[2]:.4f},  # D
    {best_params[3]:.4f},  # FF
    {best_params[4]:.4f},  # NL
    {best_params[5]:.4f},  # V
    {best_params[6]:.4f},  # bias
])
""")


if __name__ == "__main__":
    main()

