from math import log10 as log
class Log_Double:
    def __init__(self, val=0.0):
        if isinstance(val, Log_Double):
            self.is_zero = val.is_zero
            self.val = val.val
        else:
            self.is_zero = True if val == 0.0 else False
            self.val = 0.0 if val == 0.0 else log(val)

    def __eq__(self, other):
        return Log_Double(other).val == self.val

    def __add__(self, other):
        out = Log_Double(other)
        if self.is_zero:
            return out
        if out.is_zero:
            return self
        if self.val > out.val:
            out.val = log(1 + 10**(out.val - self.val)) + self.val
        else:
            out.val += log(1 + 10**(self.val - out.val))
        return out

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        out = Log_Double(other)
        print(out)
        if self.is_zero:
            out.val = 0.0
            out.is_zero = True
            return out
        if out.is_zero:
            return out
        if self.val > out.val:
            out.val = log(1 - 10**(out.val - self.val)) + self.val
        elif self.val < out.val:
            out.val = log(1 - 10 ** (self.val - out.val)) + out.val
        else:
            out.is_zero = True
            out.val = 0.0
        return out

    def __isub__(self, other):
        other = Log_Double(other)
        if self.is_zero:
            return self
        if other.is_zero:
            return other
        if self.val > other.val:
            self.val += log(1 - 10**(other.val - self.val))
        elif self.val < other.val:
            t = other
            other = self
            self = t
        else:
            self.is_zero = True
            self.val = 0.0
        return self

    def __mul__(self, other):
        out = Log_Double(other)
        if out.is_zero or self.is_zero:
            out.val = 0.0
            out.is_zero = True
            return out
        out.val += self.val
        return out

    def __imul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = Log_Double(other)
        if self.is_zero or other.is_zero:
            return Log_Double()
        out = Log_Double(self)
        out.val -= other.val
        return out

    def __itruediv__(self, other):
        return self.__truediv__(other)

    def __lt__(self, other):
        other = Log_Double(other)
        if other.is_zero:
            return False
        if self.is_zero:
            return True
        return self.val < other.val

    def __gt__(self, other):
        out = Log_Double(other)
        if out.is_zero:
            return True
        if self.is_zero:
            return False
        return self.val > out.val

    def __radd__(self, other):
        out = Log_Double(other)
        if self.is_zero:
            return out
        if out.is_zero:
            return self
        if self.val > out.val:
            out.val = log(1 + 10**(out.val - self.val)) + self.val
        else:
            out.val += log(1 + 10**(self.val - out.val))
        return out

    def __str__(self):
        return str(self.val)

    def __pow__(self, power, modulo=None):
        out = Log_Double(self)
        if out.is_zero or self.is_zero:
            out.val = 0.0
            out.is_zero = True
            return out
        out.val = power * out.val
        return out

    def get_float(self):
        if self.is_zero:
            return 0
        return 10**self.val


def split_data(data, num_splits):
    n = len(data)
    splitted_data = []
    shift = n // num_splits
    for i in range(num_splits):
        splitted_data.append(data[i * shift:i * shift + shift])
    if n % num_splits != 0:
        splitted_data.append(data[num_splits * shift:num_splits * shift + n % num_splits])
    return splitted_data
