"""importing packages"""
import multiprocessing as mp
from multiprocessing import Pool
import functools
import time
import csv


def time_me(func):
    """function to time how long other functions take to process"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Ran {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def fibo(fibo_input):
    """function to calculate fibo values"""
    if fibo_input <= 1:
        return 1
    return fibo(fibo_input - 1) + fibo(fibo_input - 2)


@time_me
def calc_for_serial(one_range):
    """standard function to calculate fibo value for range of values"""
    results = []
    for num in one_range:
        results.append([fibo(i) for i in range(num)])
    return results


@time_me
def calc_for_parallel(one_range):
    """function to calculate fibo value for range of values using parallel"""
    results = [fibo(i) for i in range(one_range)]
    return results


@time_me
def do_parallel(runs):
    """function to execute fibo using pool size dependent on # of elements"""
    size = len(runs)
    # results = []
    with Pool(size) as pool:
        [results] = [pool.map(calc_for_parallel, runs)]
    return results

@time_me
def do_parallel_2(runs):
    """function to execute fibo using pool dependent on # of available cpu"""
    pool_size = mp.cpu_count()
    pool_size = mp.Pool(pool_size)
    [results] = [pool_size.map(calc_for_parallel, runs)]
    return results

@time_me
def do_serial(runs):
    """function to execute fibo not using parallel processing"""
    results = []
    results.append(calc_for_serial(runs))
    [results] = results #pylint: disable=W0632
    return results


@time_me
def save_serial(results, header, footer):
    """function to save results of do serial"""
    counter = 0
    for result_set in results:
        for file in result_set:
            counter += 1
            with open('res' + str(counter) + '.txt', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([header, file, footer])

@time_me
def save_parallel(results, header, footer): #pylint: disable=W0613
    """function to save results of do parallel"""
    print("Nothing was saved")

@time_me
def save_parallel_2(runs):
    """function to save results of parallel_2"""
    pool_size = mp.cpu_count()
    pool_size = mp.Pool(pool_size)
    pool_size.map(save_serial, runs)


@time_me
def main():
    """main function to execute script"""
    ranges = ((10, 12, 5, 7, 4, 3, 2), (35, 30, 25, 40), (40, 38, 42))
    parallel = []
    parallel2 = []
    serial = []
    for number_set in ranges:
        parallel.append(do_parallel(number_set))
        serial.append(do_serial(number_set))
        parallel2.append(do_parallel_2(number_set))

    header = "h" * 100
    footer = "f" * 100

    parallel = parallel[0]
    serial = serial[0]
    save_serial(serial, header, footer)
    save_parallel(parallel, header, footer)
    save_parallel_2(parallel2)


if __name__ == "__main__":
    main()
