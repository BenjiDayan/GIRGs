import csv
import multiprocessing
import random
import time
import queue

class Temp:

    def __init__(self):
        self._dict_queue = multiprocessing.Manager().Queue()
        self.resultspath = 'temp_results.csv'

    def listener(self, q):
        with open(self.resultspath, "w") as results_file:
            wrote_header = False
            while True:
                result_dict = q.get()
                print(f'got: {result_dict}')
                if result_dict is None:
                    print('Done writing to csv')
                    break
                else:
                    print(f'Saving csv of result {result_dict["Graph"]}, {result_dict["Number"]}')
                    # results_file.write(str(result_dict) + '\n')
                    # continue
                
                all_keys = set(result_dict.keys())
                fieldnames = sorted(all_keys)

                dict_writer = csv.DictWriter(results_file, fieldnames)
                if not wrote_header:
                    dict_writer.writeheader()
                    wrote_header = True

                dict_writer.writerow(result_dict)
                results_file.flush()

    def do(self, x):
        val = random.random()
        my_dict = {'Graph': random.choice('abc'), 'Number': val}
        self._dict_queue.put(my_dict)
        print(my_dict)
        time.sleep(val)


    def execute(self):
        writer_pool = multiprocessing.Pool(1)
        out = writer_pool.apply_async(self.listener, (self._dict_queue,), error_callback=custom_callback)

        pool = multiprocessing.Pool(3)
        pool.map(self.do, list(range(15)))

        pool.close()
        pool.join()

        self._dict_queue.put(None)

        out.get()
        writer_pool.close()
        writer_pool.join()



def custom_callback(error):
	print(error, flush=True)


if __name__ == '__main__':
    t = Temp()
    t.execute()