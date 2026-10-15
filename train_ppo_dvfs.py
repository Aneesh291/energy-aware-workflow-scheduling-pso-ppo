from stable_baselines3 import PPO
from core.node import ResourceNode
from core.dag_parser import load_dag_from_json
from models.pso_scheduler import run_pso
from models.rl_scheduler_env import RLTaskSchedulerEnv

WORKFLOW = "workflows/workflow_50.json"

def create_nodes():
    return [
        ResourceNode("Edge1", "edge", 1.2, 10, 25),
        ResourceNode("Fog1", "fog", 2.5, 20, 40),
        ResourceNode("Cloud1", "cloud", 3.6, 30, 60)
    ]

G = load_dag_from_json(WORKFLOW)
tasks = [G.nodes[t] for t in G.nodes]

# Step 1 — PSO baseline
pso_nodes = create_nodes()
best_assignment, _ = run_pso(tasks, pso_nodes)

# Step 2 — DVFS-only training
nodes = create_nodes()

env = RLTaskSchedulerEnv(
    tasks,
    nodes,
    enable_migration=False,
    pso_initial_assignment=best_assignment
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=15000)
model.save("ppo_scheduler_dvfs")

print("PPO_DVFS training complete.")