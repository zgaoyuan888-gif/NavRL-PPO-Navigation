#!/usr/bin/env python3
"""Generate verified NavRL homework report - corrected reward function."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import os

OUT = '/home/user/Doubao/chats/10006173868345602'
CHART = os.path.join(OUT, 'navrl_charts')

doc = Document()
for s in doc.sections:
    s.top_margin = Cm(2.2); s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.5); s.right_margin = Cm(2.5)

st = doc.styles['Normal']
st.font.name = 'Calibri'; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn('w:eastAsia'), 'Calibri')

NAVY = RGBColor(0x1E, 0x3A, 0x5F)
DARK = RGBColor(0x2B, 0x24, 0x20)
GRAY = RGBColor(0x5A, 0x50, 0x48)
GOLD = RGBColor(0xB8, 0x7A, 0x2E)

def sr(run, size=11, bold=False, color=DARK):
    run.font.name = 'Calibri'; run.font.size = Pt(size)
    run.font.bold = bold; run.font.color.rgb = color

def title(t, sub):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2); sr(p.add_run(t), 19, True, NAVY)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(16); sr(p.add_run(sub), 11, color=GRAY)

def h1(t):
    p = doc.add_paragraph(); p.paragraph_format.space_before = Pt(14); p.paragraph_format.space_after = Pt(4)
    sr(p.add_run(t), 14, True, NAVY)

def body(t, bold=False, color=DARK):
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(3); p.paragraph_format.line_spacing = 1.4
    sr(p.add_run(t), 11, bold, color)

def bullet(t):
    p = doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after = Pt(2)
    sr(p.add_run(t), 11)

def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers)); t.style = 'Light Grid Accent 1'
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ''
        sr(c.paragraphs[0].add_run(h), 10.5, True, RGBColor(0xFF,0xFF,0xFF))
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ''
            sr(cells[i].paragraphs[0].add_run(str(v)), 10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def add_image(path, width=6.0, caption=None):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(path, width=Inches(width))
    if caption:
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.paragraph_format.space_after = Pt(10)
        sr(c.add_run(caption), 9.5, color=GRAY)

# ===== CONTENT =====
title('NavRL Homework Report (Verified)',
      'Autonomous Robot Navigation with PPO · Reward Function & Results')

# 1
h1('1. System Overview')
body('This project evaluates a pre-trained PPO-based NavRL policy for autonomous 2D robot navigation. '
     'The robot uses LiDAR-style ray-cast sensing to reach a goal while avoiding static and dynamic obstacles. '
     'We use the provided pre-trained checkpoint and run evaluation; no retraining is performed.')
body('Network: CNN (LiDAR) + MLP (state/dynamic obstacles) + Beta actor (continuous velocity output).', bold=True)

# 2
h1('2. State and Action Space')
add_table(
    ['Component', 'Shape', 'Description'],
    [
        ['robot_state', '8-dim', 'Relative goal position (unit vec + distance) + robot velocity in goal frame'],
        ['lidar / ray_cast', '1 × 36 × 4', '360° scan at 10° resolution, 4 vertical angles, max range 4.0 m'],
        ['dynamic_obstacle', '1 × 5 × 10', 'Up to 5 nearby agents: relative pos, distance, velocity, size'],
        ['direction', '1 × 3', 'Unit vector from robot to goal'],
    ]
)
body('Action: continuous 3D velocity (vx, vy, vz), clipped to ±2.0 m/s, transformed from local to world frame. '
     'dt = 0.1 s in evaluation; max 600 steps (60 s) per episode.')

# 3 - CORRECTED REWARD FUNCTION
h1('3. Reward Function (Verified from env.py)')
body('The reward is a dense per-step continuous reward, NOT sparse terminal +10/-10. '
     'Source: isaac-training/training/scripts/env.py lines 547–580.', bold=True)

add_table(
    ['Component', 'Formula', 'Purpose', 'Weight'],
    [
        ['reward_vel', 'v · (goal−pos) / |goal−pos|', 'Rewards moving toward goal', '1.0'],
        ['Baseline', 'constant 1.0', 'Per-step positive base reward', '1.0'],
        ['reward_safety_static', 'mean(log(lidar_range − lidar_scan))', 'Encourages distance from static obstacles', '1.0'],
        ['reward_safety_dynamic', 'mean(log(dist to nearest moving obstacle))', 'Encourages distance from dynamic agents', '1.0'],
        ['penalty_smooth', '||v_t − v_{t−1}||₂', 'Penalizes jerky velocity changes', '−0.1'],
        ['penalty_height', 'height violation²', '0 in 2D ground robot (3D drone only)', '−8.0'],
    ]
)

body('Total reward formula:', bold=True)
body('reward = reward_vel + 1.0 + reward_safety_static + reward_safety_dynamic − 0.1·penalty_smooth − 8.0·penalty_height')

body('Terminal conditions:', bold=True)
add_table(
    ['Event', 'Condition', 'Note'],
    [
        ['Goal reached', 'distance < 0.5 m', 'Episode ends; no explicit +10 bonus'],
        ['Collision', 'lidar hit within 0.3 m of obstacle', '−50 penalty is commented out in code'],
        ['Timeout', 'max 2200 steps per episode', 'Training setting'],
        ['Out of bounds', 'height < 0.2 m or > 4.0 m', '3D drone boundary (N/A for 2D)'],
    ]
)

body('Note: There is no "keep-moving" reward. The agent is rewarded for moving toward the goal and staying '
     'away from obstacles, not for motion itself.', color=GRAY)

# 4
h1('4. Hyperparameters (from config files)')
add_table(
    ['Hyperparameter', 'Value', 'Source'],
    [
        ['Discount factor γ', '0.99', 'ppo.py: GAE(0.99, 0.95)'],
        ['GAE λ', '0.95', 'ppo.py'],
        ['Learning rate (all networks)', '5e-4', 'cfg/ppo.yaml'],
        ['PPO clip ratio ε', '0.1', 'cfg/ppo.yaml'],
        ['Entropy loss coefficient', '1e-3', 'cfg/ppo.yaml'],
        ['Frames per policy update', '32', 'cfg/ppo.yaml'],
        ['Training epochs per batch', '4', 'cfg/ppo.yaml'],
        ['Number of minibatches', '16', 'cfg/ppo.yaml'],
        ['Max action speed', '2.0 m/s', 'cfg/ppo.yaml'],
        ['Max episode length (training)', '2200 steps', 'cfg/train.yaml'],
        ['Total training frames', '1.2 × 10⁹', 'cfg/train.yaml'],
        ['Static obstacles per env', '350', 'cfg/train.yaml'],
        ['Dynamic obstacles per env', '80', 'cfg/train.yaml'],
        ['Number of parallel envs', '2', 'cfg/train.yaml'],
        ['Simulation dt (training)', '0.016 s', 'cfg/sim.yaml'],
    ]
)

# 5
h1('5. Three-Scenario Results (30 episodes each)')
add_table(
    ['Scenario', 'Robots', 'Success', 'Collision', 'Timeout', 'Avg Time'],
    [
        ['A: Static obstacles only', '1', '100.0%', '0.0%', '0.0%', '8.83 s'],
        ['B: Dynamic (multi-robot)', '3', '100.0%', '0.0%', '0.0%', '19.1 s'],
        ['C: Mixed (static + dynamic)', '5', '100.0%', '0.0%', '0.0%', '19.1 s'],
    ]
)

body('Observations:')
bullet('Pre-trained policy achieves 100% success across all three scenarios.')
bullet('Multi-robot scenarios take ~2× longer as robots yield and avoid each other.')
bullet('Near-miss events (passing within 0.3 m) occur frequently but no collisions.')

# 6
h1('6. Code Reference')
add_table(
    ['Component', 'File', 'Location'],
    [
        ['Reward calculation', 'isaac-training/training/scripts/env.py', 'lines 547–580'],
        ['State extraction', 'NavRL/quick-demos/utils.py', 'get_robot_state(), get_ray_cast()'],
        ['PPO algorithm', 'NavRL/quick-demos/ppo.py', 'class PPO'],
        ['Agent inference', 'NavRL/quick-demos/agent.py', 'Agent.plan()'],
        ['Obstacle generation', 'NavRL/quick-demos/env.py', 'generate_obstacles_grid()'],
        ['Hyperparameter config', 'isaac-training/training/cfg/', 'ppo.yaml, train.yaml'],
    ]
)

out = os.path.join(OUT, 'NavRL_Homework_Report_Verified.docx')
doc.save(out)
print(f'Saved: {out}')
print(f'Size: {os.path.getsize(out)} bytes')
