import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DatasetGenerator():
    def __init__(self,
                 fleet_info_fn='../../data/example_fleet_info.csv',
                 fleet_sensor_logs_fn='../../data/example_fleet_sensor_logs.csv',
                 fleet_statistics_fn='../../data/generation/fleet_statistics.csv',
                 num_vehicles=250,
                 num_sensor_readings=100,
                 period_ms=30000):
        self.statistics_df = pd.read_csv(fleet_statistics_fn)
        self.fleet_info_fn=fleet_info_fn
        self.fleet_sensor_logs_fn=fleet_sensor_logs_fn
        self.num_vehicles = num_vehicles
        self.size_per_type = int(num_vehicles/len(self.statistics_df))  # TODO: allow smaller values of num_vehicles.
        self.start_time = datetime(2020,1,1,0,0,0,0)
        self.time_delta = timedelta(microseconds=period_ms*1000)
        self.num_sensor_readings = num_sensor_readings

    def generate_dataset(self):
        fleet_info_df = self._generate_fleet_info()
        fleet_sensor_df = self._generate_sensor_logs(fleet_info_df)

        self._write_to_csv(fleet_info_df, fleet_sensor_df)

    def _generate_fleet_info(self):
        data = []
        df = self.statistics_df.drop(columns=['voltage_mean',
                                              'current_mean',
                                              'voltage_std',
                                              'current_std',
                                              'resistance_mean',
                                              'resistance_std'])
        for idx, row in df.iterrows():
            for c in range(self.size_per_type):
                vehicle_id = self.size_per_type * idx + c
                data.append([vehicle_id, row.make, row.model, row.year, row.vehicle_class, row.engine_type])
        cols = list(df.columns)
        cols.insert(0, 'vehicle_id')
        fleet_info_df = pd.DataFrame(data=data, columns=cols)

        # No need to shuffle the dataset at this point.
        #fleet_info_df.vehicle_id = fleet_info_df.vehicle_id.sample(frac=1)  #  Shuffle rows

        return fleet_info_df

    def _generate_sensor_logs(self, fleet_info_df):
        data = []
        for idx, stats in self.statistics_df.iterrows():
            threshold = SensorSeries.threshold(stats)

            for c in range(self.size_per_type):
                vehicle_id = self.size_per_type * idx + c
                timestamp = self.start_time

                sensor_series = iter(SensorSeries(stats))

                samples = []
                for t in range(self.num_sensor_readings):
                    voltage, current, resistance = next(sensor_series)
                    target = int(abs(voltage-current*resistance) > threshold)
                    samples.append(target)
                    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f")
                    data.append([vehicle_id, target, timestamp_str, voltage, current])

                    timestamp += self.time_delta
                
        fleet_sensor_logs_df = pd.DataFrame(data=data, columns=['vehicle_id', 'target', 'timestamp', 'voltage', 'current'])
        return fleet_sensor_logs_df

    def _write_to_csv(self, fleet_info_df, fleet_sensor_df):
        fleet_info_df.to_csv(path_or_buf=self.fleet_info_fn, mode='w', header=True, index=False)
        fleet_sensor_df.to_csv(path_or_buf=self.fleet_sensor_logs_fn, mode='w', header=True, index=False)

class SensorSeries:
    def __init__(self, stats):
        self.stats = stats

    def __iter__(self):
        stats = self.stats
        self.voltage_rw = iter(CappedRandomWalk(mean=stats.voltage_mean, std=stats.voltage_std))
        self.current_rw = iter(CappedRandomWalk(mean=stats.current_mean, std=stats.current_std))
        self.resistance_rw = iter(CappedRandomWalk(mean=stats.resistance_mean, std=stats.resistance_std))
        return self

    def __next__(self):
        voltage = next(self.voltage_rw)
        current = next(self.current_rw)
        resistance = next(self.resistance_rw)
        return voltage, current, resistance

    def threshold(stats):
        sensor_series = iter(SensorSeries(stats))

        samples = []
        for t in range(100):
            voltage, current, resistance = next(sensor_series)
            samples.append(abs(voltage - current * resistance))

        samples = np.array(samples)
        threshold = samples.mean() + 2 * samples.std()
        return threshold


class CappedRandomWalk:
    def __init__(self, mean, std, std_factor=.1, cap=.5):
        self.mean = mean
        self.std = std
        self.std_factor = std_factor
        self.cap = cap

    def __iter__(self):
        self.state = np.random.normal(self.mean, self.std)
        return self

    def __next__(self):
        state = self.state
        delta = np.random.normal(0, self.std*self.std_factor)
        if self.state > self.mean + self.cap:
            self.state -= abs(delta)
        elif self.state < self.mean - self.cap:
            self.state += abs(delta)
        else:
            self.state += delta

        return state


if __name__ == '__main__':
    dataset_generator = DatasetGenerator(num_sensor_readings=50)
    dataset_generator.generate_dataset()

