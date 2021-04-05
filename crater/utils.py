def tuplify(thing):
    try:
        return tuple(thing)
    except TypeError:
        return (thing,)
