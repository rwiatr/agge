import threading
import time
import sys

def print_time(thread_id, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print(f'{thread_id}: {time.ctime(time.time())}')

thread_list = []


for i in range(int(sys.argv[1])):
    try:
        thread_list += [threading.Thread(target=print_time, args=(i, 2))]
    except:
        print(f"unable to start create a thread {i}")

for id, thread in enumerate(thread_list):
    try:
        thread.start()
    except:
        print(f'error in thread {id}')

for thread in enumerate(thread_list):
    thread.join()

