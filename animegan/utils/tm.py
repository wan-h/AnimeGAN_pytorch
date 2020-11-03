# coding: utf-8
# Author: wanhui0729@gmail.com

import time
import datetime

class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        '''time start'''
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=False):
        '''time stop'''
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        if average:
            self.average_time = self.total_time / self.calls
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

def generate_time_str(the_time=None, tag=''):
    '''
    generate time string
    :param the_time: time.time()
    :return: time string
    '''
    the_time = the_time or time.time()
    time_str = str(int(the_time * 10000000))
    if tag:
        time_str = time_str + '_' + tag
    return time_str

def generate_datetime_str(the_time=None, formate='%Y-%m-%d %H:%M:%S', tag=''):
    '''
    generate datetime string
    :param the_time: datetime.datetime()
    :param formate: datetime string format
    :return: datetime string
    '''
    the_time = the_time or datetime.datetime.now()
    time_str = the_time.strftime(formate)
    if tag:
        time_str = time_str + '_' + tag
    return time_str