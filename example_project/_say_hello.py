class SayHello:
    """A class that says hello to the person of your choice."""

    def __init__(self, name):
        self.name = name

    def say_hello(self):
        print(f"Hello, {self.name}!")
