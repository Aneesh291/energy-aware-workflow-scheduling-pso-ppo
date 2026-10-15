import numpy as np
from stable_baselines3 import PPO
from models.rl_scheduler_env import RLTaskSchedulerEnv
from models.pso_scheduler import run_pso
from core.node import ResourceNode
from core.dag_parser import load_dag_from_json
from engine.task_migrator import check_and_migrate

# Load trained PPO model (trained with PSO warm start, Discrete(3) action space)
model = PPO.load("ppo_scheduler")

# Load workflow
G = load_dag_from_json("workflows/workflow_50.json")
tasks = [G.nodes[task] for task in G.nodes]

# Create nodes
def create_nodes():
    nodes = [
        ResourceNode("Edge1", "edge", 1.2, 10, 25),
        ResourceNode("Fog1", "fog", 2.5, 20, 40),
        ResourceNode("Cloud1", "cloud", 3.6, 30, 60)
    ]
    for n in nodes:
        n.enable_dvfs = True
    return nodes

# Step 1 — Run PSO to get initial assignment (matches training setup)
pso_nodes = create_nodes()
best_assignment, _ = run_pso(tasks, pso_nodes, num_particles=10, num_iterations=15)

# Step 2 — Create env with PSO assignment (matches training: Discrete(3) action space)
nodes = create_nodes()
env = RLTaskSchedulerEnv(
    tasks,
    nodes,
    enable_migration=True,
    pso_initial_assignment=best_assignment
)
obs, _ = env.reset()

total_energy = 0.0
assignments = []

print("\n📡 PPO Smart Scheduler — Task Assignments:")
while True:
    action, _states = model.predict(obs, deterministic=True)

    # pso_initial_assignment mode: action is a single DVFS index (Discrete(3))
    node_index = best_assignment[env.current_task_index]
    dvfs_index = int(np.array(action).flatten()[0])
    node = nodes[node_index]
    dvfs_mode = ["normal", "low", "boost"][dvfs_index]

    obs, reward, done, _, _ = env.step(action)
    total_energy += -reward

    task_id = env.current_task_index - 1
    assignments.append((task_id, node.id, dvfs_mode))

    print(f"Task {task_id} → {node.id} | DVFS: {dvfs_mode}")

    if done:
        break

# Post-scheduling task migration
print("\n📦 Post-Scheduling Task Migrations:")
migrations = check_and_migrate(tasks, nodes)
for task, from_node, to_node, energy in migrations:
    print(f"Moved Task (CPU={task['cpu']}) from {from_node} → {to_node}, Energy used: {energy:.2f} J")

print(f"\n⚡ PPO Final Total Energy Used: {total_energy:.2f} J")
