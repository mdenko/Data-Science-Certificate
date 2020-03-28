"""importing csv package to save as csv"""
import csv


def fibo(input_number):
    """function to calcuate fibonacci series"""
    if input_number <= 1:
        return 1
    return fibo(input_number - 1) + fibo(input_number - 2)


def save_it(alist, name):
    """function to save csv files"""
    with open(name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(alist)


def calc(write=False, ranges=(25, 35, 20, 42)):
    """function to calcuate fibo and save csv"""
    # you can only change code between this line....
    # and within fibo and save_it
    def calc_fun():
        part_d = tuple(range(ranges[3]))
        return part_d
    part_d = calc_fun()
    part_a = part_d[0:24]
    part_b = part_d[0:34]
    part_c = part_d[0:21]
    if write:
        save_it(part_a, "a.csv")
        save_it(part_b, "b.csv")
        save_it(part_c, "c.csv")
        save_it(part_d, "d.csv")

    # and this line


if __name__ == "_def __call__(self, *args):_main__":
    calc(True, (24, 34, 21, 42))
