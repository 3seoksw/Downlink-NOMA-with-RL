import os

from datetime import datetime


class Logger:
    def __init__(self, save_dir: str = "logs", save_every: int = 20):
        now = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.save_dir = os.path.join(save_dir, now)
        os.mkdir(self.save_dir)
        self.save_every = save_every
        self.log = {}
        self.curr_step = 0

    def log_step(self, value, mode: str = "train", log: str = "loss"):
        self.curr_step += 1

        log_type = mode + "_" + log
        if log_type not in self.log:
            self.log[log_type] = []

        self.log[log_type].append(value)

        if self.curr_step % self.save_every == 0:
            self.save()

    def save(self):
        for key in self.log.keys():
            idx = key.index("_") + 1
            mode = key[: idx - 1]
            log = key[idx:]

            dir_path = os.path.join(self.save_dir, mode)
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, f"{log}.txt")

            with open(file_path, "a") as f:
                for value in self.log[key]:
                    f.write(str(value) + "\n")

            self.log[key] = []
