from .noma_user import NOMA_User
#from ~~ commenv
import numpy as np


class BaseStation:
    def __init__(self, users: list[NOMA_User] = []):
        self.users = users
        self.symbols = {} # random 4 bit sequence dictionary. key : user_idx, val : rand bit
        self.multiplexed_signals = {} # key : ch, val : multiplexed signal
        self.channel_user_pair = {} # key : channel, val : (user_idx, power)

    def set_transmission_symbol(self):
        for user in self.users:
            # generate 4 bit random sequence
            random_symbol_sequence = np.random.randint(0, 2, size=4)

            # save it to dictionary symbols
            self.symbols[user.user_idx] = random_symbol_sequence



    def send_multiplexed_signals(self):
        # noma_user can use multiplexed signals by referencing multiplexed_signals dictionary.
        # ch 통해서 user 뽑는다 .. usr.set-m-signal()  multiplex signal : dictonary


    #Power can be used implicitly(by referencing) by using channel-power lookup table.
    #In here, I used channel & power explicitly
    #In test code, you will execute allocate_resource(), len(self.users) times
    def allocate_resources(self, user_idx: int, channel: int, power: float):
        self.channel_user_pair.setdefault(channel, []).append((user_idx, power))
    # to multiplex, two same channel must be picked.

    """
    Test code for allocate_resource()
    u1 => ch 3 pow 10
    self.allocate_resources(user_idx=1, channel=3, power=10)
    u2 => ch 3 pow 20
    self.allocate_resources(user_idx=2, channel=3, power=20)
    print(self.channel_user_pair)  # {3: [(1, 10), (2, 20)]}
    """

    def multiplex(self):
        for channel, user_idx_power_pairs in self.channel_user_pair.items():
            # channel : key, user_idx_power_pairs : val (user_idx, power)
            user_idx_power_pairs = list(user_idx_power_pairs)  # tuple to list

            # extract user_idx from val's 1st arg
            user_ids = [user_idx for user_idx, _ in user_idx_power_pairs]

            # extract user's symbol
            symbols = [self.symbols[user_idx] for user_idx in user_ids]

            # extract power
            powers = [power for _, power in user_idx_power_pairs]

            # multiplex
            multiplexed_signal = np.sqrt(powers[0]) * symbols[0] + np.sqrt(powers[1]) * symbols[1]

            # save multiplexed signal into multiplexed_signals dictionary
            self.multiplexed_signals[channel] = multiplexed_signal






