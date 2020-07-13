from sagemaker.source.preprocessing import pivot_data, sample_dataset
from sagemaker.source.config import Config
import time

if __name__ == '__main__':
    config = Config('config/config.yaml')
    start = time.time()
    pivot_data(config)
    sample_dataset(config)
    end = time.time()
    print('Elapsed: ', end - start, 'seconds')
