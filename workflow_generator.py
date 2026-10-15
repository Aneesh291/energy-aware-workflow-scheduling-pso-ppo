import json
import random


def generate_scientific_workflow(
        num_tasks=50,
        num_layers=5,
        cpu_range=(1, 8),
        mem_range=(256, 4096),
        runtime_range=(2, 20),
        seed=42,
        output_file="workflows/generated_workflow.json"
):
    random.seed(seed)

    tasks = []
    dependencies = []
    layers = [[] for _ in range(num_layers)]

    for i in range(num_tasks):
        layer_index = int(i * num_layers / num_tasks)
        task_id = f"T{i+1}"

        task = {
            "id": task_id,
            "cpu": random.randint(*cpu_range),
            "mem": random.randint(*mem_range),
            "runtime": random.randint(*runtime_range)
        }

        tasks.append(task)
        layers[layer_index].append(task_id)

    # Create dependencies between layers
    for i in range(num_layers - 1):
        for parent in layers[i]:
            children = random.sample(
                layers[i + 1],
                k=random.randint(1, min(3, len(layers[i + 1])))
            )
            for child in children:
                dependencies.append({
                    "from": parent,
                    "to": child
                })

    workflow = {
        "tasks": tasks,
        "dependencies": dependencies
    }

    with open(output_file, "w") as f:
        json.dump(workflow, f, indent=4)

    print(f"Workflow saved to {output_file}")


if __name__ == "__main__":
    generate_scientific_workflow(25, output_file="workflows/workflow_25.json")
    generate_scientific_workflow(50, output_file="workflows/workflow_50.json")
    generate_scientific_workflow(100, output_file="workflows/workflow_100.json")