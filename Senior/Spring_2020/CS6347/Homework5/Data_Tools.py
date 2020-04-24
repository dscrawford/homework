import numpy as np
class Dataset:
    def __init__(self, num_variables, num_rows, data):
        self.num_variables = num_variables
        self.num_rows = num_rows
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        self._i = 0
        return iter(self.data)

    def __next__(self):
        if self._i <= self.num_rows:
            self._i += 1
            return self.data[self._i]
        raise StopIteration

    def get_column(self, item):
        return self.data[:, item]

    def get_assignment_count(self, var_assign):
        vars = np.array(list(var_assign.keys()))
        count = 0
        for row in self.data:
            increment = True
            for var in vars:
                if row[var] != var_assign[var]:
                    increment = False
                    break
            count += increment
        return count

class Data_Extractor:
    def __init__(self, file_path):
        self.data = Dataset(*self.parse_file(file_path))

    def get_data(self):
        return self.data

    def parse_file(self, file_path):
        l = list(filter(None, reversed(open(file_path).read().split('\n'))))
        num_variables, num_rows = [int(i) for i in list(filter(None, l.pop().split(' ')))]
        data = np.array([[-1 if i == '?' else int(i) for i in list(filter(None, row.split(' ')))] for row in l])
        return num_variables, num_rows, data