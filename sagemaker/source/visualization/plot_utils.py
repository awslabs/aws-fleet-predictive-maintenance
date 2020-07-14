import numpy as np
import pandas as pd

def build_chart(array, min_label_percentage=0.05):
    sizes = pd.value_counts(array)
    N = array.shape[0]
    
    out_labels = []
    out_sizes = []
    others_size = 0
    for index, values in sizes.iteritems():
        if (values/N) > min_label_percentage:
            out_labels.append(index)
            out_sizes.append(values)
        else:
            others_size += values
    out_labels.append("Others")
    out_sizes.append(others_size)
    
    out_percentage = (np.array(out_sizes)/N).tolist()
    return out_labels, out_percentage