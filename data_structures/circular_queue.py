# Circular Queue class
class CircularQueue:

    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def enqueue(self, inp):
        if self.size()+1 <= self.max_size:
          self.queue.append(inp)
        else:
          self.dequeue()
          self.queue.append(inp)

    def dequeue(self):
        self.queue.pop(0)

    def front(self):
        return self.queue[0]

    def rear(self):
        return self.queue[-1]

    def size(self):
        return len(self.queue)

    def sum_queue(self):
        return sum(self.queue)

    def print_queue(self):
      print(self.queue)
