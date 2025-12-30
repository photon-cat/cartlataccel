#!/bin/bash
# Run comprehensive evaluation of PID controller under different conditions

echo "=========================================="
echo "PID Controller Evaluation Suite"
echo "=========================================="

echo -e "\n1. Testing with NO noise..."
python eval_pid.py --n_rollouts 5 --noise_mode None

echo -e "\n2. Testing with REALISTIC noise..."
python eval_pid.py --n_rollouts 5 --noise_mode REALISTIC

echo -e "\n3. Testing with HIGH noise..."
python eval_pid.py --n_rollouts 5 --noise_mode HIGH

echo -e "\n4. Testing different PID gains (P=0.3, I=0.05, D=-0.1)..."
python eval_pid.py --n_rollouts 5 --p 0.3 --i 0.05 --d -0.1

echo -e "\n=========================================="
echo "Evaluation complete!"
echo "=========================================="

