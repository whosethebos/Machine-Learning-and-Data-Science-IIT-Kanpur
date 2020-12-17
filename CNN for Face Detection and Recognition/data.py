import numpy as np

def triplet_generator():

    while True:
        anchor_batch = np.random.rand(4, 96, 96, 3)
        pos_batch = np.random.rand(4, 96, 96, 3)
        neg_batch = np.random.rand(4, 96, 96, 3)
        yield [anchor_batch , pos_batch, neg_batch], None
