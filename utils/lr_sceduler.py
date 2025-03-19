class LambdaLinearScheduler:
    def __init__(self, optimizer, warm_up_steps=10000, cycle_lengths=1e12, f_start=1e-6, f_max=1.0, f_min=1.0):
        self.optimizer = optimizer
        self.warm_up_steps = warm_up_steps
        self.cycle_lengths = cycle_lengths  # Large number to prevent decay
        self.f_start = f_start
        self.f_max = f_max
        self.f_min = f_min
        self.step_num = 0

    def get_lr(self):
        """Compute the current learning rate."""
        if self.step_num < self.warm_up_steps:
            # Linear warm-up from f_start to f_max
            return self.f_start + (self.f_max - self.f_start) * (self.step_num / self.warm_up_steps)
        else:
            # Maintain f_min (constant learning rate)
            return self.f_min

    def step(self):
        """Update the learning rate in the optimizer."""
        self.step_num += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
