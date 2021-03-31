import heapq


# A priority queue which allows custom sorting function
# The main priority is the 'f' value. if two elements have the same f value- then sort by creation time.
# lastly sort by the action priority as defined in the exercise
class PriorityQueue:
    # constructor
    def __init__(self, f=lambda x: x):
        self.heap = []
        self.f = f
        self.actions_priority = {'R': 2, 'RD': 3, 'D': 4, 'LD': 5, 'L': 6, 'LU': 7, 'U': 8, 'RU': 9}
        self.prev_parent = None
        self.counter = 0

    # append an element to the queue
    def append(self, item):
        if (not item.action):
            action_priority = 1
        else:
            action_priority = self.actions_priority[item.action]
        heapq.heappush(self.heap, (self.f(item), self.counter, action_priority, item))
        # only increase the counter if the elements don't have the same parent
        if (item.parent != self.prev_parent):
            self.counter += 1
        self.prev_parent = item.parent

    # append all given items to the queue
    def extend(self, items):
        for item in items:
            self.append(item)

    # pop an element from the queue(with the highest priority)
    def pop(self):
        if self.heap:
            return heapq.heappop(self.heap)[3]
        else:
            raise Exception('Trying to pop from empty PriorityQueue.')

    # returns the len of the queue
    def __len__(self):
        return len(self.heap)

    # defines how instances of class behave when they appear at right side of 'in' and 'not in' operator
    def __contains__(self, key):
        return any([item == key for _, _, _, item in self.heap])

    # get an item using a key
    def __getitem__(self, key):
        for value, _, _, item in self.heap:
            if item == key:
                return value
        raise KeyError(str(key) + " is not in the priority queue")

    # delete an item using a key
    def __delitem__(self, key):
        try:
            del self.heap[[item == key for _, _, _, item in self.heap].index(True)]
        except ValueError:
            raise KeyError(str(key) + " is not in the priority queue")
        heapq.heapify(self.heap)

    # returns a string representation
    def __repr__(self):
        return str(self.heap)
