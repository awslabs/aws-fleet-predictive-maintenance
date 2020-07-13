import yaml
import pandas as pd

class Config:
    def __init__(self, filename, fetch_sensor_headers=True):
        with open(filename) as config_stream:
            self.config_yaml = yaml.safe_load(config_stream)
        for k, v in self.config_yaml.items():
            self.__dict__[k] = v
        if fetch_sensor_headers:
            self.sensor_headers = self.get_sensor_headers()

    def __repr__(self):
        output_string = "Config properties\n"
        for k, v in self.config_yaml.items():
            output_string += "{}={}\n".format(k, v)
        return output_string

    def get_sensor_headers(self):
        df = pd.read_csv(self.fleet_sensor_logs_fn, chunksize=1)
        df_chunk = next(iter(df))
        sensor_headers = df_chunk.columns.tolist()
        sensor_headers.remove(self.vehicle_id_column)
        sensor_headers.remove(self.timestamp_column)
        sensor_headers.remove(self.target_column)

        return sensor_headers