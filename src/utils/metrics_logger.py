import wandb
import matplotlib.pyplot as plt


class MetricsLogger:
    def __init__(self, prefix=None, log_epoch=True):
        self.prefix = prefix
        self.iteration_metrics = []
        self.running_stats = {}
        self.it_counter = {}
        self.stats = {}
        self.log_epoch = log_epoch
        self.log_dict = {}
        self.epoch = 1

    def add_media_metric(self, name):
        wandb.define_metric(name, step_metric=self.apply_prefix('epoch'))
        self.log_dict[self.apply_prefix(name)] = None

    def log_image(self, name, image, caption=None):
        self.log_dict[self.apply_prefix(name)] = wandb.Image(image, caption=caption)

    def log_plot(self, name):
        wandb.log({self.apply_prefix(name): plt})

    def apply_prefix(self, name):
        return f'{self.prefix}/{name}' if self.prefix is not None else name

    def add(self, name, iteration_metric=False):
        # initialize per-epoch stats
        self.stats[name] = []
        wandb.define_metric(self.apply_prefix(name),
                            step_metric=self.apply_prefix('epoch'))
        self.log_dict[self.apply_prefix(name)] = None

        if iteration_metric:
            # track per-iteration metrics
            self.iteration_metrics.append(name)
            self.stats[f'{name}_per_it'] = []
            self.running_stats[name] = 0.0
            self.it_counter[name] = 0

    def update_it_metric(self, name, value):
        # called each iteration
        self.running_stats[name] += value
        self.it_counter[name] += 1

    def update_epoch_metric(self, name, value, prnt=False):
        # called when you have a direct epoch-level value
        self.stats[name].append(value)
        self.log_dict[self.apply_prefix(name)] = value
        if prnt:
            print(name, "=", value)

    def finalize_epoch(self):
        # compute & log averages of iteration metrics
        for name in self.iteration_metrics:
            total = self.running_stats.get(name, 0.0)
            count = self.it_counter.get(name, 1)
            if count == 0:
                epoch_value = 0.0
            else:
                epoch_value = total / count
            self.stats[name].append(epoch_value)
            self.log_dict[self.apply_prefix(name)] = epoch_value
            if self.log_epoch:
                print(name, "=", epoch_value)

            # reset counters for next epoch
            self.running_stats[name] = 0.0
            self.it_counter[name] = 0

        # log epoch number
        self.log_dict[self.apply_prefix('epoch')] = self.epoch
        self.epoch += 1

        # push to W&B
        wandb.log(self.log_dict, commit=True)

    def last(self, metric: str) -> float:
        """
        Return the most recent value logged under `metric`.
        """
        if metric not in self.stats or len(self.stats[metric]) == 0:
            raise RuntimeError(f"No entries logged for metric '{metric}' yet.")
        return self.stats[metric][-1]
