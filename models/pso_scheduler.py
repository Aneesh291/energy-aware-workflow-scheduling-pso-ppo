import random

class Particle:
    def __init__(self, num_tasks, num_nodes):
        # Random assignment of tasks to nodes
        self.position = [random.randint(0, num_nodes - 1) for _ in range(num_tasks)]
        self.velocity = [0] * num_tasks
        self.best_position = list(self.position)
        self.best_fitness = float('inf')
        self.fitness = float('inf')
from core.node import ResourceNode

def evaluate_fitness(particle, tasks, nodes):
    # Reset node state before evaluation
    for node in nodes:
        node.tasks = []
        node.apply_dvfs("normal")  # Reset DVFS mode at the start

    total_energy = 0.0

    for task_index, node_index in enumerate(particle.position):
        task = tasks[task_index]
        node = nodes[node_index]

        # PSO evaluates at normal frequency — DVFS is PPO's responsibility at runtime
        node.apply_dvfs("normal")

        energy = node.execute_task(task["cpu"], task["runtime"])

        if task["cpu"] > 2 and node.cpu < 1.5:
            energy += 10

        total_energy += energy

    return total_energy


def run_pso(tasks, nodes, num_particles=10, num_iterations=30):
    num_tasks = len(tasks)
    num_nodes = len(nodes)

    swarm = [Particle(num_tasks, num_nodes) for _ in range(num_particles)]

    # Global best across all particles
    global_best_position = None
    global_best_fitness = float('inf')

    w = 0.5      # inertia weight
    c1 = 1.0     # cognitive (self)
    c2 = 1.5     # social (swarm)

    for iteration in range(num_iterations):
        for particle in swarm:
            fitness = evaluate_fitness(particle, tasks, nodes)
            particle.fitness = fitness

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = list(particle.position)

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = list(particle.position)

        # Update velocities and positions
        for particle in swarm:
            for i in range(num_tasks):
                r1, r2 = random.random(), random.random()

                # Basic discrete update
                new_velocity = (w * particle.velocity[i] +
                                c1 * r1 * (particle.best_position[i] - particle.position[i]) +
                                c2 * r2 * (global_best_position[i] - particle.position[i]))
                particle.velocity[i] = int(new_velocity)

                # Update position by "shifting" the assignment
                particle.position[i] = (particle.position[i] + particle.velocity[i]) % num_nodes

    print("\nBest assignment found:", global_best_position)
    print("Minimum energy used:", global_best_fitness)
    return global_best_position, global_best_fitness
