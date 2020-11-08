# coding: utf-8
# Author: wanhui0729@gmail.com

'''
DatasetInfo class
'''

class DatasetInfo(object):
    '''
    manage dataset
    '''
    def __init__(self, factory, **kwargs):
        '''
        Arguments:
            dataDir: dataset dir
            factory: factory of data interface
            split: train or test
            kwargs: other args for data interface
        '''
        self.factory = factory
        self.kwargs = kwargs

    def get(self):
        '''
        return:
            dict contain the infomation of dataset
        '''
        return dict(
            factory=self.factory,
            args=self.kwargs
        )
