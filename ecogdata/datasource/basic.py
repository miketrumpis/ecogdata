import numpy as np


class ElectrodeDataSource(object):
    """
    a parent class for all types
    """
    pass

class PlainArraySource(ElectrodeDataSource):

    """
    Will include in-memory data arrays from a raw data source. I.e. primary file(s) have been loaded and scaled and
    possibly filtered in arrays described here.

    """
    pass