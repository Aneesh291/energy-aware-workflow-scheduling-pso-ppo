class ResourceNode:
    def __init__(self, node_id, layer, cpu_ghz, power_idle, power_full, supports_dvfs=True):
        self.id = node_id
        self.layer = layer

        self.base_cpu = cpu_ghz
        self.base_power_full = power_full
        self.power_idle = power_idle

        self.supports_dvfs = supports_dvfs
        self.enable_dvfs = True

        self.cpu = cpu_ghz
        self.dvfs_mode = "normal"
        self.tasks = []

    def apply_dvfs(self, mode):
        if not self.supports_dvfs or not self.enable_dvfs:
            self.cpu = self.base_cpu
            self.dvfs_mode = "normal"
            return

        self.dvfs_mode = mode

        if mode == "low":
            self.cpu = self.base_cpu * 0.6
        elif mode == "boost":
            self.cpu = self.base_cpu * 1.3
        else:
            self.cpu = self.base_cpu

    def execute_task(self, task_cpu, base_runtime):
        """
        base_runtime = workload complexity (seconds at 1 GHz baseline)
        execution_time scales inversely with CPU frequency
        """

        # Execution time scales inversely with CPU frequency
        execution_time = base_runtime / self.cpu

        # Realistic dynamic power scaling (cubic with frequency)
        freq_ratio = self.cpu / self.base_cpu
        dynamic_power = self.base_power_full * (freq_ratio ** 3)

        # Total power
        total_power = self.power_idle + dynamic_power

        # Energy
        energy_used = total_power * execution_time

        self.tasks.append({
            "cpu": task_cpu,
            "time": execution_time,
            "energy": energy_used,
            "mode": self.dvfs_mode
        })

        return energy_used

    def __str__(self):
        return f"{self.id} ({self.layer}) - CPU: {self.cpu:.2f}GHz, Mode: {self.dvfs_mode}"