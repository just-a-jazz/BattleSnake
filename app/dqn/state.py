class State(object):
    def __init__(self, world):
        self._populate_state(world)

    def _populate_state(self, world):
        self.snakes = {snake['id'] : [(pos['x'], pos['y'])
                        for pos in snake['body']['data']]
                        for snake in world['snakes']['data']}

        self.food = [(pos['x'], pos['y']) for pos in world['food']['data']]

        self.grid_size = (world['width'], world['height'])

        self.health = world['you']['health']
