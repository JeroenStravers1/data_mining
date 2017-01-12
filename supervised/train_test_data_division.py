import random
import datetime
import os

class DataDivider:

    _DEFAULT_TRAINING_DATA_PERCENTAGE = 0.33
    _MIN_SEED = 0
    _MAX_SEED = 9999
    _CURRENT_LOCATION = ""
    _READ_ONLY = "r"

    _RAW_DATA_HEADER_ROWS = 1

    _DATAFILE_TRAIN = "train"
    _DATAFILE_TEST = "test"
    _DATAFILE_EXTENSION = ".csv"

    def __init__(self, input_file_location, output_file_location=None, seed=None, train_data_percentage=None):
        self._input_file_location = input_file_location
        self._output_location = output_file_location if output_file_location is not None \
            else self._CURRENT_LOCATION
        self._train_data_percentage = train_data_percentage if train_data_percentage is not None \
            else self._DEFAULT_TRAINING_DATA_PERCENTAGE
        self._seed = seed if seed is not None \
            else random.randint(self._MIN_SEED, self._MAX_SEED)

    def generate_train_test_files(self):
        read_data = self._get_data_from_file()
        train_file, test_file = self._generate_data_files()
        self._fill_train_test_files(train_file, test_file, read_data)

    def _get_data_from_file(self):
        with open(self._input_file_location, "r") as raw_data:
            read_data = raw_data.readlines()
            return read_data

    def _generate_data_files(self):
        timestamp = datetime.datetime.utcnow()
        formatted_timestamp = timestamp.isoformat().replace(":", ".")
        train_data_filename = self._generate_file_name(formatted_timestamp, self._DATAFILE_TRAIN)
        test_data_filename = self._generate_file_name(formatted_timestamp, self._DATAFILE_TEST)
        self._create_empty_file(train_data_filename)
        self._create_empty_file(test_data_filename)
        return train_data_filename, test_data_filename

    def _generate_file_name(self, timestamp, data_file_type):
        data_file_identity = "_" + str(self._seed) + "_" + self._DATAFILE_EXTENSION
        return timestamp + "_" + self._output_location + data_file_type + data_file_identity

    def _create_empty_file(self, file_name):
        with open(file_name, "w") as current_file:
            pass

    def _fill_train_test_files(self, train_file, test_file, read_data):
        self._write_header_rows(train_file, test_file, read_data)

    def _write_header_rows(self, train_file, test_file, read_data):
        for header_data_row in range(self._RAW_DATA_HEADER_ROWS):
            with open(train_file, "a") as train_data:
                train_data.write(read_data[header_data_row])
            with open(test_file, "a") as test_data:
                test_data.write(read_data[header_data_row])

    def _divide_data_rows_in_train_and_test_files(self):



        """generate list of train row indices (based on seed)
        sort this list ascending
        loop over dataset:
            if index == indices[current]: row -> traindata
            else row -> testdata
            current++
        #clear read_data variable -->> discard references to this object, let GC sort em out
        #clear indices list
        """


"""
get data as rows
make traincsv, testcsv with name == train_ + seed + _ + timestamp + .csv
use seed -> populate train.csv, test.csv
"""

if __name__ == "__main__":
    dv = DataDivider("mushrooms.csv")
    dv.generate_train_test_files()
    print(dv._seed)
    print(dv._train_data_percentage)

    #random.seed(6543)
    #for i in range(10):
    #    print(random.randint(1, 10))






