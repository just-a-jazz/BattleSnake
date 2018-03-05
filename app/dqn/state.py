class State(object):
    def __init__(self, world):
        self._populate_state(world)

    def _populate_state(self, world):
        self.self_id = world['you']['id']
        self.self_health = world['you']['health']

        self.grid_size = (world['width'], world['height'])

        self.snakes = {snake['id'] : [(pos['x'], pos['y'])
                        for pos in snake['body']['data']]
                        for snake in world['snakes']['data']}

        self.food = [(pos['x'], pos['y']) for pos in world['food']['data']]

    def get_self_snake_coords(self):
        return self.snakes[self.self_id]

    def is_new_game(self):
        return len(set(self.snakes[self.self_id])) == 1
