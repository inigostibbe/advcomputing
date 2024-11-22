import multiprocessing
import time

def cpu_bound(number):
    return sum(i * i for i in range(number))

def find_sums(numbers,numthreads):
    with multiprocessing.Pool(numthreads) as pool:
        pool.map(cpu_bound, numbers)

if __name__ == "__main__":
    worksize = 200 # number of parallel calculations
    numbers = [500000 + x for x in range(worksize)]
    maxthreads=multiprocessing.cpu_count()
    for numthreads in range(1,maxthreads+1):
        start_time = time.time()
        find_sums(numbers,numthreads)
        duration = time.time() - start_time
        print(f"Duration {duration} seconds for {numthreads} threads")
