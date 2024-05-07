# from .decoder import Decoder


class NOMA_User:
    def __init__(self, id=123):
        self.user_id = id
        # self.decoder = Decoder(idx=self.user_id)
        self.signals = []
        self.multiplexed_signals = []

        self.open_channel()

    def open_channel(self):
        """Constantly receives signals from Base Station (BS)"""

    def set_signals(self, signals: list[float]) -> bool:
        if signals is None or len(signals) == 0:
            return False

        self.signals = signals
        return True

    def sic_and_decode(self, signals: list[float]) -> list[float]:
        """Run Successive Interference Cancellation (SIC)"""
        return signals
