WHITE_COLOR = {'r': 255, 'g': 255, 'b': 255}
BLACK_COLOR = {'r': 0, 'g': 0, 'b': 0}


def draw_text(location, text, color=WHITE_COLOR):
    return {'type': 'text', 'x': location.x, 'y': location.y, 'text': text,
            'r': color['r'], 'g': color['g'], 'b': color['b']}


def draw_line(start, end, color=WHITE_COLOR):
    return {'type': 'line', 'start_x': start.x, 'start_y': start.y, 'start_z': start.z,
            'end_x': end.x, 'end_y': end.y, 'end_z': end.z,
            'r': color['r'], 'g': color['g'], 'b': color['b']}


def draw_circle(location, radius=25, color=WHITE_COLOR):
    return {'type': 'circle', 'x': location.x, 'y': location.y, 'z': location.z,
            'radius': radius, 'r': color['r'], 'g': color['g'], 'b': color['b']}
