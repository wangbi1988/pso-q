import multiprocessing as mp
import logging


def bad_test_func(ii):
    print('Calling bad function with arg %i' % ii)
    name = "file_%i.log" % ii
    logging.basicConfig(filename=name, level=logging.DEBUG)
    if ii < 4:
        log = logging.getLogger()
    else:
        log = "Test log %i" % ii
    return log


def good_test_func(ii):
    print('Calling good function with arg %i' % ii)
    instance = ('hello', 'world', ii)
    return instance


def pool_test(func):
    def callback(item):
        print('This is the callback')
        print('I have been given the following item: ')
        print(item)

    num_processes = 3
    pool = mp.Pool(processes=num_processes)
    results = []
    for i in range(5):
        res = pool.apply_async(func, (exampleJob(i),), callback=callback)
        #        res = pool.apply_async(func, (i,), callback=callback)
        results.append(res)
    pool.close()
    pool.join()


class exampleJob():
    def __init__(self, name):
        self.name = name;

    def run(self, ii=None):
        import time;
        import numpy as np;
        time.sleep(np.random.rand())
        return self.name, 'completed', {'name': self.name}, ii;


def main():
    print('#' * 30)
    print('Calling pool test with bad function')
    print('#' * 30)

    pool_test(bad_test_func)

    print('#' * 30)
    print('Calling pool test with good function')
    print('#' * 30)
    pool_test(good_test_func)


if __name__ == '__main__':
    main()
