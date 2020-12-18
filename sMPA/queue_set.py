#coding=UTF-8
"""
This code provides a series of utilities for queue operation
"""

# Author: Liang Chen <squirrel_d@126.com>
# License: BSD 3 clause
class Queue_Set(object):
    def __init__(self):
        self.queue = []
        self.size = 0
    def put(self,i):
        '''
        add an element into the queue

        Args:
            i: element of any type

        Returns:
            boolean: indicates whether this operation is successfully done
        '''
        if i in self.queue:
            return False
        else:
            self.queue.insert(0,i)
            self.size = len(self.queue)
            return True
    def get(self):
        '''
        pop an element from the queue

        Returns:
            elment: an element of any type
        '''
        element = self.queue.pop()
        self.size = len(self.queue)
        return element
    def isEmpty(self):
        '''
        judge if the queue is empty

        Returns:
            boolean:
        '''
        if self.size==0:
            return True
        else:
            return False
    def getSize(self):
        '''
        return size of queue

        Returns:
            int:
        '''
        return self.size
        