# Queue class
class Queue:

    def __init__(self):
        self.queue = []

    def enqueue(self, inp):
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
