from users.noma_user import NOMA_User


class BaseStation:
    def __init__(self, users: list[NOMA_User] = []):
        self.users = users
        self.signals = []
        self.multiplexed_signals = []

    def send_signals(self):
        for usr in self.users:
            usr.set_signals(self.signals)
            success = usr.sic_and_decode()

    def allocate_resources(self, user_idx: int, channel: float, power: float):
        self.users[user_idx].set_signals(channel, power)
