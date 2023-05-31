import numpy as np
import multiprocessing
import time
import os

# https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
# https://pythonspeed.com/articles/faster-multiprocessing-pickle/

var_dict = {}
def init_worker(X, X_shape):
    var_dict['X'] = X
    var_dict['X_shape'] = X_shape


def worker_func(i):
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    time.sleep(1)
    out = np.asscalar(np.sum(X_np[i, :]))
    temp = X_np[0, 0]
    X_np[0, 0] = i
    return i, out, temp, os.getpid()

if __name__ == '__main__':
    X_shape = (50, 1000)
    data = np.arange(50*1000).reshape(50, 1000)
    X = multiprocessing.RawArray('d', 50*1000)
    X_np = np.frombuffer(X, dtype=np.float64).reshape(X_shape)
    np.copyto(X_np, data)

    start_time = time.time()
    with multiprocessing.Pool(processes=20, initializer=init_worker, initargs=(X, X_shape)) as pool:
        # result = pool.map(worker_func, range(X_shape[0]))
        # print('Results (pool): \n', np.array(result))

        results = pool.imap_unordered(worker_func, range(X_shape[0]))
        for i, out, temp, name in results:
            print(f'i: {i}, name: {name}, out: {out}, temp: {temp}')
    print(f'Time: {time.time() - start_time}')

