# Stack class in Python
class Stack:

    def __init__(self):
        self.stack = ()

    def push(self, element):
        self.stack += (element,)

    def top(self):
        return self.stack[-1]

    def pop(self):
        pop_item = self.stack[-1]
        self.stack = self.stack[:-1]
        return pop_item

    def size(self):
        return len(self.stack)

    def empty(self):
        if len(self.stack) == 0:
            return True
        else:
            return False

    def print_stack(self):
        print(self.stack)

    def reverse(self):
      reverse_stack = ()
      while self.size() > 0:
        popped_item = self.pop()
        reverse_stack += (popped_item,)
      return reverse_stack

    def min(self):
      minimum = min(self.stack)
      return minimum

    def max(self):
      maximum = max(self.stack)
      return maximum
