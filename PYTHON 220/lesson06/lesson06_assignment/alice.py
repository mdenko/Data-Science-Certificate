"""
performance analysis
"""
import csv

class Speedy:
    """Class to execute functions for alice"""
    def __init__(self, func):
        self.func = func
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.func(*args)
        return self.memo[args]

    def fibo(self):
        """function to calculate Fibonacci sequence"""
        if self <= 1:
            return 1
        return self - 1 + self - 2


    def save_it(self, name):
        """function to savec csv files"""
        with open(name, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self)


    def calc(write=False, ranges=(25, 35, 20, 42)): #pylint: disable=E0213
        """function to calculate fibo and save files"""
        part_a = [Speedy.fibo(i) for i in range(ranges[0])]
        if write:
            Speedy.save_it(part_a, "a.csv")
        part_b = [Speedy.fibo(i) for i in range(ranges[1])]
        if write:
            Speedy.save_it(part_b, "b.csv")
        part_c = [Speedy.fibo(i) for i in range(ranges[2])]
        if write:
            Speedy.save_it(part_c, "c.csv")
        part_d = [Speedy.fibo(i) for i in range(ranges[3])]
        if write:
            Speedy.save_it(part_d, "d.csv")

if __name__ == "__main__":
    Speedy.calc(True, (24, 34, 21, 42))
