"""value type modeling the probablistic measures on the program context"""
from typing import Optional
from dataclasses import dataclass
from termcolor import colored

from rinarak.utils import stprint_str
from .dsl_types import ListType

class ProbValue(object):
    def __init__(self, vtype, value, prob = 1.0):
        self.vtype = vtype
        self.value = value
        self.prob = prob
    
    def __repr__(self):
        return (colored("Value", "cyan", attrs=["bold"]) + 
                colored(":[", "white") + 
                colored(f"{self.value}", "blue") + 
                colored("]-", "white") + 
                colored(f"{self.vtype}", "cyan") + " " +
                colored("P", "cyan", attrs=["bold"]) + 
                colored(":[", "white") + 
                colored(f"{self.prob}", "blue") + 
                colored("]", "white"))

class Value(ProbValue):
    def __init__(self, vtype, value):
        super().__init__(vtype, value, prob = 1.0)

class ListValue(Value):
    """A list of values."""

    dtype: ListType
    """The type of the value."""

    @property
    def element_type(self):
        return self.dtype.element_type

    values: tuple
    """The values."""

    def __init__(self, dtype: ListType, values: Sequence[Any]):
        """Initialize the Value object.

        Args:
            dtype: the type of the value.
            values: the values.
        """
        super().__init__(dtype)
        self.values = tuple(values)

    def __len__(self):
        return len(self.values)

    def __str__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        return f'{{{elements_str}}}'

    def __repr__(self):
        elements_str = ', '.join(str(v) for v in self.values)
        return f'ListValue({{{elements_str}}}, dtype={self.dtype})'

"""the tensor state of measurement and otherwise constructed"""

QINDEX = slice(None)

class ObjectType(object):
    pass

@dataclass
class StateObjectReference(object):
    name : str
    idx : int
    dtype: Optional[ObjectType] = None

    def clone(self, dtype: Optional[ObjectType] = None) -> 'StateObjectReference':
        return StateObjectReference(self.name, self.index, dtype or self.dtype)


class TensorValue(ProbValue):
    pass

class ProbTensorValue(ProbValue):
    pass