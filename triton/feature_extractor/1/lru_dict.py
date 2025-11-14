from collections import OrderedDict


class LRUDict(OrderedDict):
    """
    LRU字典
    """

    def __init__(self, max_length):
        super().__init__()
        self.max_length = max_length

    def __setitem__(self, key, value):
        if len(self) >= self.max_length:
            self.popitem(last=False)
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
