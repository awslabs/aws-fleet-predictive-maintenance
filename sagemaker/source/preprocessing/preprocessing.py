from collections import OrderedDict
import pandas as pd
from sklearn.utils import resample
from .dataframewriter import DataFrameWriter

def pivot_data(config):
    """
    Take fleet info and fleet sensor log files and combine them into a dataset. Each example in the data set is a
    window of sensor readings with meta data columns and a target column.

    This method writes data to an output location specified by the configuration.

    Args:
        config: Configuration object.
    """
    fleet_info = pd.read_csv(config.fleet_info_fn)
    fleet_sensor_logs = pd.read_csv(config.fleet_sensor_logs_fn,
                                    chunksize=config.chunksize)  # Support potentially large sensor logs.
    
    dataset_writer = DataFrameWriter(filename=config.fleet_dataset_fn, chunksize=config.processing_chunksize)

    for chunk_idx, sensor_df in enumerate(fleet_sensor_logs):
        print("Processing Sensor Data Chunk {}".format(chunk_idx + 1))

        # Convert timestamp column to have correct datatype.
        sensor_df[config.timestamp_column] = pd.to_datetime(sensor_df[config.timestamp_column])
        sensor_df.sort_values([config.vehicle_id_column, config.timestamp_column], inplace=True)
        sensor_columns = sensor_df.columns.drop(labels=[config.vehicle_id_column,
                                                        config.target_column,
                                                        config.timestamp_column])

        for row_idx, row in sensor_df.iterrows():
            time_start = row[config.timestamp_column]
            time_end = time_start + pd.Timedelta(config.period_ms, unit='ms') * config.window_length
            vehicle_id = row[config.vehicle_id_column]
            interval_filter = (sensor_df[config.vehicle_id_column] == vehicle_id) & \
                              (sensor_df[config.timestamp_column] < time_end) & \
                              (sensor_df[config.timestamp_column] >= time_start)
            sample = sensor_df[interval_filter]

            if len(sample) == config.window_length:
                target = sample[config.target_column].iloc[0]

                # Notes: This can be done more efficiently.
                inst = OrderedDict()
                # TODO: The order of the columns seems to be the same....
                inst[config.vehicle_id_column] = vehicle_id
                inst[config.period_column] = config.period_ms
                inst[config.target_column] = target
                inst[config.timestamp_column] = time_start
                for k, v in fleet_info.iloc[vehicle_id].iteritems():
                    inst[k] = v

                for col in sensor_columns.values:
                    for i in range(config.window_length):
                        inst[col + '_' + str(i)] = sample[col].iloc[i]

                dataset_writer.append(inst)

    dataset_writer.flush_buffer()
    print('Wrote fleet dataset (unsampled) to file.')

def sample_dataset(config, train_ratio=0.8):
    dataset_df = pd.read_csv(config.fleet_dataset_fn, chunksize=config.chunksize)
    train_writer = DataFrameWriter(filename=config.train_dataset_fn, chunksize=config.processing_chunksize)
    test_writer = DataFrameWriter(filename=config.test_dataset_fn, chunksize=config.processing_chunksize)

    for chunk_idx, chunk_df in enumerate(dataset_df):
        print("Processing Fleet Dataset Chunk {}".format(chunk_idx + 1))
        # Split into test and train set with 50/50 probability and random sample

        # An assumption is being made here that failures occur with lower frequency. This may not be true.
        # This should be fixed.
        df_neg = chunk_df[chunk_df[config.target_column] == 0]
        df_pos = chunk_df[chunk_df[config.target_column] == 1]

        if len(df_neg) == 0 or len(df_pos) == 0:
            print("Dropping chunk because data is all one class.")
            continue

        # Down-sample negative cases.
        df_neg = resample(df_neg,
                          replace=False,
                          n_samples=len(df_pos))

        train_pos,train_neg = df_pos.sample(frac=train_ratio) , df_neg.sample(frac=train_ratio)
        test_pos, test_neg = df_pos.drop(train_pos.index), df_neg.drop(train_neg.index)

        train_writer.append(train_pos)
        train_writer.append(train_neg)
        test_writer.append(test_pos)
        test_writer.append(test_neg)

    train_writer.flush_buffer()
    test_writer.flush_buffer()

