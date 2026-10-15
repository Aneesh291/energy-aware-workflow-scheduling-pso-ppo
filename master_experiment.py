import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from core.node import ResourceNode
from core.dag_parser import load_dag_from_json
from models.pso_scheduler import run_pso
from models.rl_scheduler_env import RLTaskSchedulerEnv
from stable_baselines3 import PPO


# ===============================
# CONFIGURATION
# ===============================
WORKFLOWS = [
    "workflows/workflow_25.json",
    "workflows/workflow_50.json",
    "workflows/workflow_100.json"
]

NUM_TRIALS = 5


# ===============================
# NODE CREATION
# ===============================
def create_nodes(enable_dvfs=True):
    nodes = [
        ResourceNode("Edge1", "edge", 1.2, 10, 25),
        ResourceNode("Fog1", "fog", 2.5, 20, 40),
        ResourceNode("Cloud1", "cloud", 3.6, 30, 60)
    ]
    for n in nodes:
        n.enable_dvfs = enable_dvfs
    return nodes


# ===============================
# MODEL 1 — PSO ONLY
# ===============================
def run_pso_only(tasks):
    nodes = create_nodes(enable_dvfs=False)
    _, energy = run_pso(tasks, nodes,
                        num_particles=10,
                        num_iterations=15)
    return energy


# ===============================
# MODEL 2 — PPO ONLY (Full Control)
# ===============================
def run_ppo_only(tasks):
    nodes = create_nodes(enable_dvfs=False)

    env = RLTaskSchedulerEnv(
        tasks,
        nodes,
        enable_migration=False,
        pso_initial_assignment=None
    )

    model = PPO.load("ppo_scheduler_full")

    obs, _ = env.reset()
    done = False
    total_energy = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_energy += -reward

    return total_energy


# ===============================
# MODEL 3 — PPO + DVFS
# ===============================
def run_ppo_dvfs(tasks):
    nodes = create_nodes(enable_dvfs=True)

    env = RLTaskSchedulerEnv(
        tasks,
        nodes,
        enable_migration=False,
        pso_initial_assignment=None
    )

    model = PPO.load("ppo_scheduler_full")

    obs, _ = env.reset()
    done = False
    total_energy = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_energy += -reward

    return total_energy


# ===============================
# MODEL 4 — TRUE HYBRID
# ===============================
def run_true_hybrid(tasks):

    # Step 1 — PSO baseline node assignment
    pso_nodes = create_nodes(enable_dvfs=False)
    best_assignment, _ = run_pso(tasks, pso_nodes,
                                 num_particles=10,
                                 num_iterations=15)

    # Step 2 — PPO DVFS refinement
    nodes = create_nodes(enable_dvfs=True)

    env = RLTaskSchedulerEnv(
        tasks,
        nodes,
        enable_migration=True,
        pso_initial_assignment=best_assignment
    )

    model = PPO.load("ppo_scheduler_dvfs")

    obs, _ = env.reset()
    done = False
    total_energy = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_energy += -reward

    return total_energy


# ===============================
# EXPERIMENT LOOP
# ===============================
results = []

for workflow_path in WORKFLOWS:

    print(f"\nRunning {workflow_path}")

    G = load_dag_from_json(workflow_path)
    tasks = [G.nodes[t] for t in G.nodes]

    pso_list = []
    ppo_list = []
    ppo_dvfs_list = []
    hybrid_list = []

    for trial in range(NUM_TRIALS):

        pso_list.append(run_pso_only(tasks))
        ppo_list.append(run_ppo_only(tasks))
        ppo_dvfs_list.append(run_ppo_dvfs(tasks))
        hybrid_list.append(run_true_hybrid(tasks))

        print(f"Trial {trial+1} complete")

    results.append({
        "Workflow": workflow_path.split("/")[-1],
        "PSO_Mean": np.mean(pso_list),
        "PPO_Mean": np.mean(ppo_list),
        "PPO_DVFS_Mean": np.mean(ppo_dvfs_list),
        "Hybrid_Mean": np.mean(hybrid_list)
    })


# ===============================
# SAVE RESULTS
# ===============================
df = pd.DataFrame(results)
df.to_csv("results/four_model_comparison.csv", index=False)

print("\nFinal Comparison Table:")
print(df)


# ===============================
# PLOT RESULTS
# ===============================
workflow_sizes = [25, 50, 100]

plt.figure()
plt.plot(workflow_sizes, df["PSO_Mean"], marker='o', label="PSO Only")
plt.plot(workflow_sizes, df["PPO_Mean"], marker='s', label="PPO Only")
plt.plot(workflow_sizes, df["PPO_DVFS_Mean"], marker='^', label="PPO + DVFS")
plt.plot(workflow_sizes, df["Hybrid_Mean"], marker='D', label="True Hybrid")

plt.xlabel("Workflow Size")
plt.ylabel("Mean Energy (J)")
plt.title("Four-Model Energy Comparison")
plt.legend()
plt.grid(True)
plt.savefig("results/four_model_comparison.png")
plt.show()