import numpy as np
from stable_baselines3 import PPO
from core.node import ResourceNode
from core.dag_parser import load_dag_from_json
from models.pso_scheduler import run_pso
from models.rl_scheduler_env import RLTaskSchedulerEnv


WORKFLOW_PATH = "workflows/workflow_50.json"


def create_nodes():
    nodes = [
        ResourceNode("Edge1", "edge", 1.2, 10, 25),
        ResourceNode("Fog1", "fog", 2.5, 20, 40),
        ResourceNode("Cloud1", "cloud", 3.6, 30, 60)
    ]
    for n in nodes:
        n.enable_dvfs = True
    return nodes


# Load workflow
G = load_dag_from_json(WORKFLOW_PATH)
tasks = [G.nodes[t] for t in G.nodes]


# Step 1 — Run PSO baseline
pso_nodes = create_nodes()
best_assignment, _ = run_pso(tasks, pso_nodes,
                             num_particles=10,
                             num_iterations=15)


# Step 2 — Create Hybrid Training Environment
nodes = create_nodes()

env = RLTaskSchedulerEnv(
    tasks,
    nodes,
    enable_migration=False,           # Disable migration during training
    pso_initial_assignment=best_assignment
)


# Step 3 — Train PPO for DVFS-only control
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=256,
    batch_size=64,
    ent_coef=0.01
)

model.learn(total_timesteps=10000)

model.save("ppo_scheduler")

print("Hybrid PPO training complete.")