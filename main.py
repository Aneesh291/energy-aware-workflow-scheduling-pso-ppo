from core.dag_parser import load_dag_from_json
from core.node import ResourceNode
from models.pso_scheduler import Particle, evaluate_fitness, run_pso
from engine.task_migrator import check_and_migrate

# ---------------------------------------
# 1. Load DAG
# ---------------------------------------
G = load_dag_from_json("workflows/workflow_50.json")
print("DAG Loaded:", G is not None)
print("Number of Tasks:", len(G.nodes()))
print("Tasks:")
for node in G.nodes(data=True):
    print(node)

print("\nDependencies:")
for edge in G.edges():
    print(edge)

tasks = [G.nodes[task] for task in G.nodes]

# ---------------------------------------
# 2. Simulate Resource Nodes (Edge, Fog, Cloud)
# ---------------------------------------
nodes = [
    ResourceNode("Edge1", "edge", 1.2, 10, 25),
    ResourceNode("Fog1", "fog", 2.5, 20, 40),
    ResourceNode("Cloud1", "cloud", 3.6, 30, 60)
]

print("\nAvailable Resource Nodes:")
for node in nodes:
    print(node)

# ---------------------------------------
# 3. Optional: Manually test task execution (can remove later)
# ---------------------------------------
energy = nodes[0].execute_task(task_cpu=1.0, base_runtime=5)
print(f"\nManual Test: Executed task on {nodes[0].id}, energy used: {energy:.2f} J")

# ---------------------------------------
# 4. Optional: Evaluate one random PSO particle
# ---------------------------------------
particle = Particle(num_tasks=len(tasks), num_nodes=len(nodes))
fitness = evaluate_fitness(particle, tasks, nodes)
print("\nRandom Particle Assignment:", particle.position)
print("Total Energy Used (Fitness):", fitness)

# ---------------------------------------
# 5. Run PSO Scheduling
# ---------------------------------------
best_assignment, best_energy = run_pso(tasks, nodes, num_particles=10, num_iterations=20)

print("\n🧠 Final PSO Task-to-Node Assignment:")
for i, node_index in enumerate(best_assignment):
    print(f"Task {list(G.nodes)[i]} → {nodes[node_index].id}")

print(f"\n⚡️ PSO: Minimum Total Energy Used: {best_energy:.2f} J")

# ---------------------------------------
# 6. Apply Task Migration After PSO
# ---------------------------------------
print("\n📦 Task Migrations (After PSO Scheduling):")
migrations = check_and_migrate(tasks, nodes)

for task, from_node, to_node, new_energy in migrations:
    print(f"Moved Task (CPU={task['cpu']}) from {from_node} → {to_node}, Energy used: {new_energy:.2f} J")
