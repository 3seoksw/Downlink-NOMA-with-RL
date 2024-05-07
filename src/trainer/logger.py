import os
import pickle


class Logger:
    def __init__(self, save_dir: str = "logs", save_every: int = 20):
        self.save_dir = save_dir
        self.save_every = save_every
        self.log = {}
        self.curr_step = 0

    def log_step(self, value, mode: str = "train", log: str = "loss"):
        self.curr_step += 1

        log_type = mode + "_" + log
        if self.log.get(log_type) is None:
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

            with open(f"{dir_path}/{log}.pkl", "wb") as f:
                pickle.dump(self.log[key], f)
