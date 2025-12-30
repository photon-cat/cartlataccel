import json
import sys
import os

def analyze_benchmark(log_dir):
  """Analyze and visualize benchmark results"""
  
  # Load benchmark log
  log_path = os.path.join(log_dir, "benchmark_log.json")
  with open(log_path, 'r') as f:
    data = json.load(f)
  
  print("=" * 80)
  print("TRAINING BENCHMARK ANALYSIS")
  print("=" * 80)
  print()
  
  # Key observations
  print("## KEY FINDINGS:\n")
  
  # 1. Best checkpoint
  best_idx = min(range(len(data)), key=lambda i: data[i]['total_cost'])
  best = data[best_idx]
  worst_idx = max(range(len(data)), key=lambda i: data[i]['total_cost'])
  worst = data[worst_idx]
  
  print(f"1. BEST Performance (lowest total cost):")
  print(f"   - Checkpoint: {best['timesteps']:,} steps")
  print(f"   - Total cost: {best['total_cost']:,.2f}")
  print(f"   - Lataccel cost: {best['lataccel_cost']:.2f}")
  print(f"   - Jerk cost: {best['jerk_cost']:,.2f}")
  print(f"   - Reward: {best['reward']:.3f}")
  print()
  
  print(f"2. WORST Performance (highest total cost):")
  print(f"   - Checkpoint: {worst['timesteps']:,} steps")
  print(f"   - Total cost: {worst['total_cost']:,.2f}")
  print(f"   - This is {worst['total_cost']/best['total_cost']:.1f}x worse than best")
  print()
  
  # 3. Training progression
  initial = data[0]
  final = data[-1]
  
  print("3. TRAINING PROGRESSION:")
  print(f"   Initial (untrained) → 10k steps:")
  print(f"     Total cost: {initial['total_cost']:,.2f} → {data[1]['total_cost']:,.2f}")
  print(f"     Improvement: {((data[1]['total_cost']/initial['total_cost']-1)*100):+.1f}%")
  print()
  print(f"   Initial → Final (100k steps):")
  print(f"     Reward improved: {initial['reward']:.3f} → {final['reward']:.3f} ({((final['reward']/initial['reward']-1)*100):+.1f}%)")
  print(f"     BUT total cost got WORSE: {initial['total_cost']:,.2f} → {final['total_cost']:,.2f}")
  print(f"     Lataccel cost improved: {initial['lataccel_cost']:.2f} → {final['lataccel_cost']:.2f} ({((final['lataccel_cost']/initial['lataccel_cost']-1)*100):+.1f}%)")
  print(f"     Jerk cost EXPLODED: {initial['jerk_cost']:.2f} → {final['jerk_cost']:,.2f}")
  print()
  
  # 4. Problem analysis
  print("4. PROBLEM IDENTIFIED:")
  print("   ⚠️  PPO learns to minimize position error (reward) but ignores jerk!")
  print("   ⚠️  The agent makes very aggressive, jerky movements")
  print("   ⚠️  Jerk cost increased by ~126,000x from initial to final")
  print("   ⚠️  This confirms PPO needs jerk penalty in the reward function")
  print()
  
  # 5. Comparison with PID
  print("5. COMPARISON WITH PID:")
  print(f"   PPO best (10k steps):   total_cost = {best['total_cost']:,.2f}")
  print(f"   PID baseline:            total_cost = 2,080.39")
  print(f"   → PID is {best['total_cost']/2080.39:.1f}x better than PPO's best checkpoint")
  print()
  
  # 6. Recommendations
  print("6. RECOMMENDATIONS:")
  print("   ✓ Use the 10k step checkpoint (best performance)")
  print("   ✓ Add jerk penalty to reward: reward -= alpha * jerk^2")
  print("   ✓ Add action smoothness penalty: reward -= beta * (action_t - action_{t-1})^2")
  print("   ✓ Reduce entropy coefficient (currently 0.01) to encourage less exploration")
  print("   ✓ Consider using a different algorithm (SAC, TD3) with continuous action smoothing")
  print()
  
  print("=" * 80)
  print()
  
  # Detailed table
  print("DETAILED RESULTS BY CHECKPOINT:")
  print()
  print("Steps   | Total Cost  | Lataccel | Jerk Cost   | Reward  | vs Best | vs PID")
  print("-" * 80)
  for entry in data:
    steps = entry['timesteps']
    total = entry['total_cost']
    lat = entry['lataccel_cost']
    jerk = entry['jerk_cost']
    reward = entry['reward']
    vs_best = total / best['total_cost']
    vs_pid = total / 2080.39
    print(f"{steps:>7,} | {total:>11,.0f} | {lat:>8.2f} | {jerk:>11,.0f} | {reward:>7.3f} | {vs_best:>6.1f}x | {vs_pid:>5.0f}x")
  
  print("=" * 80)
  
  # Create simple CSV
  csv_path = os.path.join(log_dir, "results.csv")
  with open(csv_path, 'w') as f:
    f.write("timesteps,total_cost,lataccel_cost,jerk_cost,reward\n")
    for entry in data:
      f.write(f"{entry['timesteps']},{entry['total_cost']:.2f},{entry['lataccel_cost']:.2f},")
      f.write(f"{entry['jerk_cost']:.2f},{entry['reward']:.3f}\n")
  
  print(f"\nCSV exported to: {csv_path}")

if __name__ == "__main__":
  if len(sys.argv) > 1:
    log_dir = sys.argv[1]
  else:
    # Find most recent benchmark
    logs_dir = "logs"
    benchmarks = [d for d in os.listdir(logs_dir) if d.startswith("benchmark_")]
    if not benchmarks:
      print("No benchmark logs found!")
      sys.exit(1)
    log_dir = os.path.join(logs_dir, sorted(benchmarks)[-1])
  
  print(f"Analyzing: {log_dir}\n")
  analyze_benchmark(log_dir)

