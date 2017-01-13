import random
import datetime


class TrainAndTestSetGenerator:

    _DEFAULT_TRAINING_DATA_PERCENTAGE = 0.33
    _MIN_SEED = 0
    _MAX_SEED = 10000

    _RAW_DATA_HEADER_ROWS = 1

    _DATAFILE_FOLDER = "generated_sets/"
    _DATAFILE_TRAIN = "train"
    _DATAFILE_TEST = "test"
    _DATAFILE_EXTENSION = ".csv"

    _FIRST = 0
    _NEXT = 1

    def __init__(self, input_file_location, seed=None, train_data_percentage=None, output_file_location=None):
        self._input_file_location = input_file_location
        self._train_data_percentage = train_data_percentage if train_data_percentage is not None \
            else self._DEFAULT_TRAINING_DATA_PERCENTAGE
        self._seed = seed if seed is not None \
            else random.randint(self._MIN_SEED, self._MAX_SEED)
        self._output_location = output_file_location if output_file_location is not None \
            else self._DATAFILE_FOLDER

    def generate_train_test_sets(self):
        """generate a training set and a test set, using and storing the seed and timestamp in the file names"""
        extracted_data = self._get_data_from_file()
        train_file, test_file = self._generate_data_files()
        self._fill_train_test_files(train_file, test_file, extracted_data)

    def _get_data_from_file(self):
        """store the dataset in a list
        :returns list(data_rows)"""
        with open(self._input_file_location, "r") as raw_data:
            read_data = raw_data.readlines()
            return read_data

    def _generate_data_files(self):
        """name and create test and train set files
        :returns file names"""
        timestamp = datetime.datetime.utcnow()
        formatted_timestamp = timestamp.isoformat().replace(":", ".")
        train_data_filename = self._generate_file_name(formatted_timestamp, self._DATAFILE_TRAIN)
        test_data_filename = self._generate_file_name(formatted_timestamp, self._DATAFILE_TEST)
        self._create_empty_file(train_data_filename)
        self._create_empty_file(test_data_filename)
        return train_data_filename, test_data_filename

    def _generate_file_name(self, timestamp, data_file_type):
        """:returns the name for train/test set files; contains timestamp, type, seed"""
        data_file_identity = "_" + str(self._seed) + self._DATAFILE_EXTENSION
        return self._output_location + timestamp + "_" + data_file_type + data_file_identity

    def _create_empty_file(self, file_name):
        """create an empty file"""
        with open(file_name, "w") as current_file:
            pass

    def _fill_train_test_files(self, train_file, test_file, extracted_data):
        """divide the original dataset between train and test sets"""
        self._write_header_rows(train_file, test_file, extracted_data)
        train_set_indices = self._generate_train_row_indices_from_seed(extracted_data)
        self._divide_data_rows_in_train_and_test_files(train_set_indices, train_file, test_file, extracted_data)

    def _write_header_rows(self, train_file, test_file, extracted_data):
        """write the header row to the train and test data files"""
        for header_data_row in range(self._RAW_DATA_HEADER_ROWS):
            with open(train_file, "a") as train_data:
                train_data.write(extracted_data[header_data_row])
            with open(test_file, "a") as test_data:
                test_data.write(extracted_data[header_data_row])

    def _generate_train_row_indices_from_seed(self, extracted_data):
        """select indices of rows from the dataset to be used in the training set
        :returns sorted list(training_set_indices) (ascending)"""
        random.seed(self._seed)
        total_data_rows = len(extracted_data) - self._RAW_DATA_HEADER_ROWS
        total_training_rows = round(self._DEFAULT_TRAINING_DATA_PERCENTAGE * total_data_rows)
        training_rows_indices = sorted(random.sample(range(1, total_data_rows), total_training_rows))
        return training_rows_indices

    def _divide_data_rows_in_train_and_test_files(self, train_set_indices, train_file, test_file, extracted_data):
        """use the generated list of training set row indices to divide the original dataset into train and test sets.
        Loops over the entire dataset once, comparing the current index to the list of train set indices"""
        current_selected_train_set_index = self._FIRST
        total_training_rows = len(train_set_indices)
        for data_row_index, data_row in enumerate(extracted_data):
            if data_row_index >= self._RAW_DATA_HEADER_ROWS:
                if current_selected_train_set_index < total_training_rows:
                    if data_row_index == train_set_indices[current_selected_train_set_index]:
                        with open(train_file, "a") as training_set:
                            training_set.write(data_row)
                            current_selected_train_set_index += self._NEXT
                    else:
                        with open(test_file, "a") as test_set:
                            test_set.write(data_row)
                else:
                    with open(test_file, "a") as test_set:
                        test_set.write(data_row)



        """
        generate list of train row indices (based on seed)
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
    dv = TrainAndTestSetGenerator("mushrooms.csv")
    dv.generate_train_test_sets()
    #print(dv._seed)
    #print(dv._train_data_percentage)
    #for i in range(1):
    #    print(i)
    #random.seed(6543)
    #for i in range(10):
    #    print(random.randint(1, 10))






