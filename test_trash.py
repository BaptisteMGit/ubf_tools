#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test_trash.py
@Time    :   2024/02/22 15:55:14
@Author  :   Menetrier Baptiste 
@Version :   1.0
@Contact :   baptiste.menetrier@ecole-navale.fr
@Desc    :   None
"""

# ======================================================================================================================
# Import
# ======================================================================================================================
import numpy as np
import multiprocessing


def assign_frequency_ranges(frequencies, num_workers):
    # Sort the frequencies in ascending order
    sorted_frequencies = sorted(frequencies)

    # Calculate the total number of frequencies
    total_freq_count = len(sorted_frequencies)

    # Calculate the number of frequencies each worker gets on average
    avg_freqs_per_worker = total_freq_count // num_workers

    # Initialize variables
    assigned_ranges = []
    start_idx = 0

    # Distribute frequencies to workers, ensuring decreasing subarray sizes
    for i in range(num_workers):
        # Calculate the number of frequencies for the current worker
        worker_freq_count = avg_freqs_per_worker - i

        # Adjust the number of frequencies for the last worker
        if i == num_workers - 1:
            worker_freq_count = total_freq_count - start_idx

        # Calculate the end index of the current subarray
        end_idx = min(start_idx + worker_freq_count, total_freq_count)

        # Add the current subarray to the list of assigned ranges
        assigned_ranges.append(sorted_frequencies[start_idx:end_idx])

        # Update the start index for the next subarray
        start_idx = end_idx

    return assigned_ranges


def parallel_task(freq_ranges):
    # Do something with the frequency ranges
    return np.sum(freq_ranges)


def task_wraper_queue(queue, freq_ranges):
    queue.put(parallel_task(freq_ranges))


# print(assigned_ranges)

N_CORES = 8

if __name__ == "__main__":
    # Example usage:

    num_workers = 3
    frequencies = np.arange(1, 100)
    assigned_ranges = assign_frequency_ranges(frequencies, num_workers)

    processes = []
    rets = []
    # create the queue
    queue = multiprocessing.Queue()
    for i in range(num_workers):
        process = multiprocessing.Process(
            target=task_wraper_queue, args=(queue, assigned_ranges[i])
        )
        processes.append(process)
        process.start()

    for p in processes:
        ret = queue.get()  # will block
        rets.append(ret)
    for p in processes:
        p.join()
    print(rets)
