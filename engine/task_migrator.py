def check_and_migrate(tasks, nodes, migration_threshold=1.5):
    migrations = []

    for node in nodes:
        # Iterate over a copy so we can safely remove from the original
        tasks_to_check = list(enumerate(node.tasks))
        removed_indices = []

        for i, task in tasks_to_check:
            # Use base_cpu (node tier) not live cpu (DVFS-scaled) for migration decision
            if task["cpu"] > migration_threshold and node.base_cpu < 2.0:
                # Find a better target node by base tier, not DVFS-scaled cpu
                better_nodes = [n for n in nodes if n.base_cpu > node.base_cpu and n != node]
                if better_nodes:
                    target = better_nodes[0]
                    removed_indices.append(i)
                    duration = task["cpu"] / target.base_cpu
                    energy = target.execute_task(task["cpu"], duration)
                    migrations.append((task, node.id, target.id, energy))

        # Remove migrated tasks in reverse order to preserve indices
        for i in sorted(removed_indices, reverse=True):
            node.tasks.pop(i)

    return migrations
