from tianshou.data import ReplayBuffer


class TianshouReader:
    """
    Read data from HdF5 file (in Tianshou format) and add to ReplayBuffer.
    """
    def __init__(self):
        pass

    def load(self, filename: str,) -> ReplayBuffer:
        return ReplayBuffer.load_hdf5(filename)
