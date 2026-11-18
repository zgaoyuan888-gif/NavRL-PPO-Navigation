#!/usr/bin/env python3
"""Run 3-scenario evaluation for NavRL homework."""
import os, sys, json, time
import numpy as np
import torch

DEMO_DIR = "/home/user/Doubao/chats/38449775603404546/NavRL/quick-demos"
sys.path.insert(0, DEMO_DIR)
sys.path.insert(0, "/home/user/Doubao/chats/38449775603404546/navrl_assignment/scripts")

from navrl_headless import run_single, run_multi_robot, set_seed, DEFAULT_CFG
from agent import Agent

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {dev}")
agent = Agent(device=torch.device(dev))

N_EPISODES = 30
results = {}

# ===== Scenario A: Static obstacles only =====
print("\n=== Scenario A: Static obstacles only ===")
static_results = []
for seed in range(N_EPISODES):
    r = run_single(agent, seed=seed, device=dev)
    static_results.append(r)
    if (seed+1) % 10 == 0:
        print(f"  episode {seed+1}/{N_EPISODES} done")

# ===== Scenario B: Dynamic obstacles (multi-robot = dynamic agents) =====
print("\n=== Scenario B: Dynamic obstacles (multi-robot) ===")
dynamic_results = []
for seed in range(N_EPISODES):
    r = run_multi_robot(agent, seed=seed, num_robots=3, device=dev)
    dynamic_results.append(r)
    if (seed+1) % 10 == 0:
        print(f"  episode {seed+1}/{N_EPISODES} done")

# ===== Scenario C: Mixed (multi-robot with static obstacles) =====
print("\n=== Scenario C: Mixed (multi-robot + static obstacles) ===")
mixed_results = []
for seed in range(N_EPISODES):
    r = run_multi_robot(agent, seed=seed+1000, num_robots=5, device=dev)
    mixed_results.append(r)
    if (seed+1) % 10 == 0:
        print(f"  episode {seed+1}/{N_EPISODES} done")

# ===== Compute metrics =====
def summarize_static(eps):
    n = len(eps)
    succ = sum(1 for e in eps if e["success"])
    coll = sum(1 for e in eps if e["collision"])
    timeout = sum(1 for e in eps if not e["success"] and not e["collision"])
    times = [e["completion_time"] for e in eps if e["success"]]
    dists = [e["path_length"] for e in eps if e["success"]]
    return {
        "episodes": n,
        "success_rate": round(100*succ/n, 1),
        "collision_rate": round(100*coll/n, 1),
        "timeout_rate": round(100*timeout/n, 1),
        "avg_completion_time": round(np.mean(times), 2) if times else None,
        "avg_path_length": round(np.mean(dists), 2) if dists else None,
    }

def summarize_multi(eps):
    n_ep = len(eps)
    total_robots = 0
    total_success = 0
    total_collision = 0
    total_timeout = 0
    times = []
    for e in eps:
        total_robots += e["num_robots"]
        total_success += e["num_success"]
        total_collision += e["num_collision"]
        total_timeout += (e["num_robots"] - e["num_success"] - e["num_collision"])
    return {
        "episodes": n_ep,
        "total_robots": total_robots,
        "success_rate": round(100*total_success/total_robots, 1),
        "collision_rate": round(100*total_collision/total_robots, 1),
        "timeout_rate": round(100*total_timeout/total_robots, 1),
        "avg_completion_time": round(np.mean([e["completion_time"] for e in eps]), 2),
    }

summary = {
    "static_only": summarize_static(static_results),
    "dynamic_only": summarize_multi(dynamic_results),
    "mixed": summarize_multi(mixed_results),
}

print("\n===== SUMMARY =====")
print(json.dumps(summary, indent=2))

out = "/home/user/Doubao/chats/10006173868345602/navrl_hw_results.json"
with open(out, "w") as f:
    json.dump({"summary": summary, "static_episodes": [{k:v for k,v in e.items() if k not in ("trajectory","obstacles")} for e in static_results]}, f, indent=2)
print(f"\nSaved: {out}")
