# NavRL Robot Navigation with PPO - Course Project

> Reward function aligned with NavRL paper Eq.7~12 (Xu et al., RA-L 2025)

## Project Overview
This project implements robot navigation using the PPO reinforcement learning algorithm based on the NavRL framework. We evaluate the pre-trained model over 30 test scenarios across three environments (static obstacles, dynamic obstacles, and mixed). No retraining is performed — we run inference on the provided checkpoint.

## File Structure
- `src/quick-demos/` — NavRL inference code (agent.py, env.py, ppo.py)
- `src/scripts/` — Training code (env.py with reward function, train.py)
- `src/cfg/` — Hyperparameter config files (ppo.yaml, train.yaml)
- `checkpoints/` — Pre-trained model weights (navrl_checkpoint.pt)
- `figures/` — Report charts (network, reward, results, trajectory)
- `run_hw_eval.py` — Evaluation script (30 episodes × 3 scenarios)
- `navrl_hw_results.json` — Raw evaluation results
- `NavRL_Homework_Report_Verified.docx` — Final report

## Environment
```bash
conda create -n NavRL python=3.10
conda activate NavRL
pip install torch numpy matplotlib
