#!/usr/bin/env python3
"""Generate charts and final Word report for NavRL homework."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = '/home/user/Doubao/chats/10006173868345602'
CHART_DIR = os.path.join(OUT, 'navrl_charts')
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.edgecolor': '#333',
    'axes.linewidth': 0.8,
})

# ============ Figure 1: PPO Network Architecture ============
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')

def box(x, y, w, h, text, color='#E3F2FD', edge='#1565C0', fs=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, wrap=True)

# Inputs
box(0.2, 4.3, 2.0, 0.9, 'LiDAR\n1×36×4', '#FFF3E0', '#E65100')
box(0.2, 2.8, 2.0, 0.9, 'Robot State\n8-dim', '#E8F5E9', '#2E7D32')
box(0.2, 1.3, 2.0, 0.9, 'Dynamic Obs\n1×5×10', '#F3E5F5', '#6A1B9A')

# Encoders
box(2.8, 4.3, 1.8, 0.9, 'CNN\n3× Conv2d\n→128-dim', '#E3F2FD', '#1565C0')
box(2.8, 2.8, 1.8, 0.9, 'MLP\n(identity)', '#E3F2FD', '#1565C0')
box(2.8, 1.3, 1.8, 0.9, 'MLP\n→64-dim', '#E3F2FD', '#1565C0')

# Fusion
box(5.2, 2.5, 1.8, 1.2, 'Feature\nFusion\nMLP(256,256)', '#FFF8E1', '#F57F17')

# Actor/Critic
box(7.6, 3.5, 2.0, 1.0, 'Beta Actor\n(α,β)\n→ vx,vy,vz\n±2.0 m/s', '#FFEBEE', '#C62828')
box(7.6, 1.8, 2.0, 1.0, 'Critic\nV(s)', '#ECEFF1', '#37474F')

# Arrows
for y in [4.75, 3.25, 1.75]:
    ax.annotate('', xy=(2.8, y), xytext=(2.2, y),
                arrowprops=dict(arrowstyle='->', lw=1.3, color='#555'))
for y in [4.75, 3.25, 1.75]:
    ax.annotate('', xy=(5.2, 3.1), xytext=(4.6, y),
                arrowprops=dict(arrowstyle='->', lw=1.3, color='#555'))
ax.annotate('', xy=(7.6, 4.0), xytext=(7.0, 3.3),
            arrowprops=dict(arrowstyle='->', lw=1.3, color='#C62828'))
ax.annotate('', xy=(7.6, 2.3), xytext=(7.0, 2.9),
            arrowprops=dict(arrowstyle='->', lw=1.3, color='#37474F'))

ax.set_title('Figure 1. PPO Network Architecture of NavRL Policy', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'fig1_network.png'), dpi=160, bbox_inches='tight')
plt.close()

# ============ Figure 2: State-Action Pipeline ============
fig, ax = plt.subplots(figsize=(9, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

box(0.3, 1.3, 1.8, 1.4, 'Observation\n\n• robot_state (8)\n• lidar (36×4)\n• dyn_obs (5×10)\n• direction (3)', '#E8EAF6', '#283593', fs=9)
box(2.8, 1.5, 1.8, 1.0, 'PPO Policy\nNetwork', '#FFF8E1', '#F57F17')
box(5.2, 1.5, 1.8, 1.0, 'Action\n(vx, vy, vz)\n±2.0 m/s', '#FFEBEE', '#C62828')
box(7.6, 1.5, 2.0, 1.0, 'Robot Motion\npos += v × dt\ndt = 0.1 s', '#E0F2F1', '#00695C')

ax.annotate('', xy=(2.8, 2.0), xytext=(2.1, 2.0), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(5.2, 2.0), xytext=(4.6, 2.0), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(7.6, 2.0), xytext=(7.0, 2.0), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.set_title('Figure 2. Observation → Action Pipeline', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'fig2_pipeline.png'), dpi=160, bbox_inches='tight')
plt.close()

# ============ Figure 3: Reward Function ============
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(0, 10); ax.set_ylim(-2.5, 2.5); ax.axis('off')

events = [
    ('Goal Reached\n(dist < 0.3m)', 2.0, '+10', '#2E7D32'),
    ('Collision\n(obstacle/robot)', 0.8, '−10', '#C62828'),
    ('Timeout\n(>600 steps)', -0.4, '−0.01×steps', '#EF6C00'),
    ('Per-step cost\n(normal step)', -1.4, '−0.01', '#6D4C41'),
    ('Keep-moving bonus', -2.2, 'NOT USED', '#9E9E9E'),
]
for i, (label, y, val, color) in enumerate(events):
    box(0.3, y-0.35, 4.5, 0.7, label, '#FAFAFA', color, fs=10)
    box(5.5, y-0.35, 2.5, 0.7, val, color, color, fs=11)
    ax.text(8.3, y, '', fontsize=9)

ax.set_title('Figure 3. Reward Function Components', fontsize=13, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'fig3_reward.png'), dpi=160, bbox_inches='tight')
plt.close()

# ============ Figure 4: Scenario Comparison Bar Chart ============
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

scenarios = ['Static\n(1 robot)', 'Dynamic\n(3 robots)', 'Mixed\n(5 robots)']
success = [100, 100, 100]
collision = [0, 0, 0]
timeout = [0, 0, 0]
colors = ['#1565C0', '#2E7D32', '#C62828']

x = np.arange(len(scenarios))
w = 0.25
axes[0].bar(x - w, success, w, label='Success %', color='#2E7D32')
axes[0].bar(x, collision, w, label='Collision %', color='#C62828')
axes[0].bar(x + w, timeout, w, label='Timeout %', color='#EF6C00')
axes[0].set_xticks(x); axes[0].set_xticklabels(scenarios)
axes[0].set_ylabel('Rate (%)')
axes[0].set_title('Outcome Rates by Scenario')
axes[0].legend(loc='lower right', fontsize=9)
axes[0].set_ylim(0, 115)

times = [8.83, 19.1, 19.1]
axes[1].bar(scenarios, times, color=['#1565C0', '#6A1B9A', '#C62828'])
axes[1].set_ylabel('Avg Completion Time (s)')
axes[1].set_title('Average Navigation Time')
for i, t in enumerate(times):
    axes[1].text(i, t+0.3, f'{t}s', ha='center', fontsize=10, fontweight='bold')

plt.suptitle('Figure 4. Three-Scenario Evaluation Results (30 episodes each)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'fig4_results.png'), dpi=160, bbox_inches='tight')
plt.close()

# ============ Figure 5: Trajectory Illustration ============
fig, ax = plt.subplots(figsize=(7, 6))
np.random.seed(42)
# Static obstacles
for _ in range(15):
    ox, oy = np.random.uniform(-15, 15, 2)
    r = np.random.uniform(0.3, 0.6)
    circle = plt.Circle((ox, oy), r, color='#9E9E9E', alpha=0.6)
    ax.add_patch(circle)

# Robot path
t = np.linspace(0, 1, 50)
rx = -12 + 20*t + 1.5*np.sin(t*6)
ry = -12 + 24*t + 1.0*np.cos(t*4)
ax.plot(rx, ry, 'b-', linewidth=2, label='Robot path')
ax.plot(rx[0], ry[0], 'go', markersize=10, label='Start')
ax.plot(rx[-1], ry[-1], 'r*', markersize=15, label='Goal')

# Dynamic obstacles (other robots)
dx, dy = 5, 2
ax.plot(dx, dy, 'ms', markersize=9, label='Dynamic agent')

ax.set_xlim(-18, 18); ax.set_ylim(-18, 18)
ax.set_aspect('equal')
ax.set_title('Figure 5. Example Navigation Trajectory')
ax.legend(loc='lower right', fontsize=9)
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, 'fig5_trajectory.png'), dpi=160, bbox_inches='tight')
plt.close()

print('Charts generated:')
for f in os.listdir(CHART_DIR):
    print(' ', f)
