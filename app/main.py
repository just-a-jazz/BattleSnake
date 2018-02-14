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

    head_url = '{}://{}/static/head.png'.format(
        bottle.request.urlparts.scheme,
        bottle.request.urlparts.netloc
    )

    return {
        'color': '#00FF00',
        'taunt': 'Whew',
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
        port=os.getenv('PORT', '8080'),
        debug = True)
