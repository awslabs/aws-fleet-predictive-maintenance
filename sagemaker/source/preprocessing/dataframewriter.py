from collections import OrderedDict
import numpy as np
import pandas as pd

class DataFrameWriter:
    def __init__(self, filename, chunksize):
        self.filename = filename
        self.chunksize = chunksize
        self.data = []
        self.first_write = True
        self.columns = None

    def append(self, datum):
        # For optimizations, this method does not support OrderedDict and DataFrame used interchangeably. Support
        # will be added in the future.
        if self.columns is None:
            self.columns = list(datum.keys())
        if isinstance(datum, OrderedDict):
            self.data.append(list(datum.values()))
        elif isinstance(datum, pd.DataFrame):
            if self.data == []:
                self.data = datum.values[:]
            else:
                self.data = np.append(self.data, datum.values, axis=0)
        else:
            raise Exception("Unsupported type passed to DataFrameWriter.")
        
        if len(self.data) >= self.chunksize:
            self.flush_buffer()

    def flush_buffer(self):
        df = pd.DataFrame(columns=self.columns, data=self.data)
        df.drop_duplicates(inplace=True)
        print('\tWriting {} records'.format(len(df)))
        if self.first_write:
            df.to_csv(path_or_buf=self.filename, mode='w', header=True, index=False)
            self.first_write = False
        else:
            df.to_csv(path_or_buf=self.filename, mode='a', header=False, index=False)
        self.data = []

