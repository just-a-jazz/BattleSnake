class Trainer(object):
    def __init__(self, discount_factor=0.99):
        self._discount_factor = discount_factor

    def train_network(self, reward):
        raise NotImplementedError()
