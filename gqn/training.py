import random


class Annealer(object):
    def __init__(self, init, delta, steps):
        self.init = init
        self.delta = delta
        self.steps = steps
        self.s = 0
        self.data = self.__repr__()
        self.recent = init

    def __repr__(self):
        return {"init": self.init, "delta": self.delta, "steps": self.steps, "s": self.s}

    def __iter__(self):
        return self

    def __next__(self):
        self.s += 1
        value = max(self.delta + (self.init - self.delta) * (1 - self.s / self.steps), self.delta)
        self.recent = value
        return value


def partition(images, viewpoints):
    """
    Partition batch into context and query sets.
    :param images
    :param viewpoints
    :return: context images, context viewpoint, query image, query viewpoint
    """
    # Maximum number of context points to use
    _, b, m, *x_dims = images.shape
    _, b, m, *v_dims = viewpoints.shape

    # "Squeeze" the batch dimension
    images = images.view((-1, m, *x_dims))
    viewpoints = viewpoints.view((-1, m, *v_dims))

    # Sample random number of views
    n_context = random.randint(2, m - 1)
    indices = random.sample([i for i in range(m)], n_context)

    # Partition into context and query sets
    context_idx, query_idx = indices[:-1], indices[-1]

    x, v = images[:, context_idx], viewpoints[:, context_idx]
    x_q, v_q = images[:, query_idx], viewpoints[:, query_idx]

    return x, v, x_q, v_q