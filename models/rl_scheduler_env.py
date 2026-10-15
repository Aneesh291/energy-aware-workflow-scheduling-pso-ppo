import gymnasium as gym
from gymnasium import spaces
import numpy as np
from engine.task_migrator import check_and_migrate


class RLTaskSchedulerEnv(gym.Env):

    def __init__(self, tasks, nodes,
                 enable_migration=True,
                 pso_initial_assignment=None):

        super(RLTaskSchedulerEnv, self).__init__()

        self.tasks = tasks
        self.nodes = nodes
        self.enable_migration = enable_migration
        self.pso_initial_assignment = pso_initial_assignment

        self.num_tasks = len(tasks)
        self.num_nodes = len(nodes)
        self.current_task_index = 0

        # Action Space
        if self.pso_initial_assignment is not None:
            # DVFS only mode
            self.action_space = spaces.Discrete(3)
        else:
            # Full control mode
            self.action_space = spaces.MultiDiscrete([self.num_nodes, 3])

        # Observation Space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(2 + self.num_nodes * 5,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_task_index = 0

        for node in self.nodes:
            node.tasks = []
            node.apply_dvfs("normal")

        return self._get_observation(), {}

    def _get_observation(self):

        task = self.tasks[self.current_task_index]

        obs = [
            task["cpu"],
            task["runtime"]
        ]

        for node in self.nodes:
            obs.extend([
                node.cpu,
                node.power_idle,
                node.base_power_full,
                0 if node.dvfs_mode == "normal"
                else (1 if node.dvfs_mode == "low" else 2),
                sum(t["cpu"] for t in node.tasks)
            ])

        return np.array(obs, dtype=np.float32)

    def step(self, action):

        task = self.tasks[self.current_task_index]

        # ----------------------------
        # Handle action safely
        # ----------------------------

        if self.pso_initial_assignment is not None:
            # Hybrid mode
            node_index = self.pso_initial_assignment[self.current_task_index]
            dvfs_mode_idx = int(np.array(action).flatten()[0])
        else:
            # Full PPO mode
            action = np.array(action).flatten()
            node_index = int(action[0])
            dvfs_mode_idx = int(action[1])

        node = self.nodes[node_index]

        # ----------------------------
        # Apply DVFS
        # ----------------------------
        mode = ["normal", "low", "boost"][dvfs_mode_idx]
        node.apply_dvfs(mode)

        # ----------------------------
        # Execute task
        # ----------------------------
        energy = node.execute_task(task["cpu"], task["runtime"])

        reward = -energy

        # Light load penalty
        reward -= 0.05 * sum(t["cpu"] for t in node.tasks)

        # Penalize poor DVFS choices
        if task["cpu"] <= 2 and mode == "boost":
            reward -= 3

        if task["cpu"] >= 5 and mode == "low":
            reward -= 5

        self.current_task_index += 1
        done = self.current_task_index >= self.num_tasks

        if done and self.enable_migration:
            migrated = check_and_migrate(self.tasks, self.nodes)
            reward -= len(migrated) * 2

        obs = (
            self._get_observation()
            if not done
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, reward, done, False, {}