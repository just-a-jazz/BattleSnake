import bottle
import os
import random

from dqn import Snake, State

@bottle.route('/static/<path:path>')
def static(path):
    return bottle.static_file(path, root='static/')


@bottle.post('/start')
def start():
    data = bottle.request.json

    head_url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT5aze3rPf_In2RQWTq0FspNhfEPljumfsiYXlJ5EUQD5UKZPR6'

    return {
        'color': '#000000',
        'taunt': 'Whew',
        'head_type': 'fang',
        'tail_type': 'curled',
        'head_url': head_url,
        'name': 'Root'
    }


@bottle.post('/move')
def move():
    data = bottle.request.json
    state = State(data)

    return {
        'move': Root.get_action(state),
        'taunt': 'Whew'
    }


# Expose WSGI app (so gunicorn can find it)
application = bottle.default_app()
Root = Snake()

if __name__ == '__main__':
    bottle.run(
        application,
        host=os.getenv('IP', '0.0.0.0'),
        port=os.getenv('PORT', '6464'),
        debug = True)
