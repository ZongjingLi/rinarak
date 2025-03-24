# -*- coding: utf-8 -*-
# @Author: Meleko
# @Date:   2023-10-13 13:07:56
# @Last Modified by:   Melkor
# @Last Modified time: 2023-10-19 04:42:09
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from rinarak.dklearn.nn import FCBlock
from rinarak.utils.tensor import logit
from rinarak.utils import indent_text
from rinarak.dsl.dsl_values import Value, MultidimensionalArrayInterface, TensorValueDict, QINDEX,\
    StateObjectReference, ListType, StateObjectList
from rinarak.dsl.dsl_types import AutoType, ObjectType

from termcolor import colored
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Mapping, List, Iterable, Tuple, Sequence
import re

class DSLExpressionError(Exception):
    pass

"""
an expression is the data class of tree with the following nodes.
"""

class Expression:
    def evaluate(self, inputs, executor):
        return inputs
    
    def __repr__(self):
        return f"Expression()"
        
    @staticmethod
    def parse_program_string(program):
        # Remove whitespace
        program = program.strip()
        
        # Assignment expression
        if ":=" in program:
            var_name, value_expr = program.split(":=", 1)
            return ValueAssignmentExpression(var_name.strip(), 
                                            Expression.parse_program_string(value_expr.strip()))
        
        # Function application with arguments
        function_match = re.match(r'^(\w+)\((.*)\)$', program)
        if function_match:
            func_name = function_match.group(1)
            args_str = function_match.group(2)
            
            # Parse arguments
            args = []
            if args_str.strip():
                # Track nested parentheses to split arguments correctly
                current_arg = ""
                paren_level = 0
                
                for char in args_str:
                    if char == ',' and paren_level == 0:
                        # End of an argument
                        args.append(Expression.parse_program_string(current_arg.strip()))
                        current_arg = ""
                    else:
                        if char == '(':
                            paren_level += 1
                        elif char == ')':
                            paren_level -= 1
                        current_arg += char
                
                # Add the last argument
                if current_arg:
                    args.append(Expression.parse_program_string(current_arg.strip()))
            
            return FunctionApplicationExpression(VariableExpression(func_name), args)
        
        # Constants (uppercase identifiers)
        if re.match(r'^[A-Z]{2,}$', program):
            return ConstantExpression("Any",program)
        
        # Numeric constant
        if re.match(r'^-?\d+(\.\d+)?$', program):
            return ConstantExpression("Any",float(program) if '.' in program else int(program))
        
        # Variable
        return VariableExpression(program)

class VariableExpression(Expression):
    def __init__(self, name):
        super().__init__()
        self.name = name
    
    def __repr__(self):
        return colored("Var", "cyan", attrs=["bold"]) + "(" + colored(self.name, "cyan") + ")"

class ObjectOrValueExpression(Expression):
    def __repr__(self):
        return colored("ObjectOrValue", "cyan", attrs=["dark"]) + "()"

class ValueAssignmentExpression(Expression):
    def __init__(self, var_name, value_expr):
        super().__init__()
        self.var_name = var_name
        self.value_expr = value_expr
    
    def __repr__(self):
        return (colored("Assign", "cyan", attrs=["bold"]) + "(" + 
                colored(self.var_name, "blue") + " " + 
                colored(":=", "white") + " " + 
                repr(self.value_expr) + ")")

class ConstantExpression(ObjectOrValueExpression):
    def __init__(self, ctype, const):
        super().__init__()
        self.ctype = ctype
        self.const = Value(ctype, const)
    
    def __repr__(self):
        return (colored("Const", "cyan", attrs=["bold"]) + "(" + 
                colored(f"{self.const}", "blue") + ")")

class FunctionApplicationExpression(ObjectOrValueExpression):
    def __init__(self, func, args):
        super().__init__()
        self.func = func
        self.args = args
    
    def __repr__(self):
        args_str = colored(", ", "white").join(repr(arg) for arg in self.args)
        return (colored("FuncApp", "cyan", attrs=["bold"]) + "(" + 
                colored(self.func, "blue") + ", " + 
                colored("[", "white") + args_str + colored("]", "white") + ")")


class State:
    """a state that maps a given feature name to a value"""
    def __init__(self):
        """ construct a symbolic or hybrid state
        Args:
            data: a diction that maps the data of each state to the actual value
        """

""""""

class TensorStateBase(State):
    """a state representation maps a given feature name to the tensor feature"""
    def __init__(self, data):
        super(data)
    
    @property
    def batch_dims(self) -> int: raise NotImplementedError()

    @property
    def features(self) -> MultidimensionalArrayInterface: raise NotImplementedError

    def __getitem__(self, name): return self.features[name]

    def clone(self) -> 'TensorStateBase': raise NotImplementedError()
    
    def __str__(self) -> str: raise NotImplementedError()

    def __repr__(self): return self.__str__()

class TensorState(TensorStateBase):
    """A state representation is essentially a mapping from feature names to tensors."""
    def __init__(self, features: Optional[Union[Mapping[str, Any], TensorValueDict]] = None, batch_dims: int = 0, internals: Optional[Dict[str, Any]] = None):
        """Initialize a state.

        Args:
            features: the features of the state.
            batch_dims: the number of batch dimensions.
            internals: the internal state of the state.
        """

        if features is None:
            features = dict()
        if internals is None:
            internals = dict()

        if isinstance(features, TensorValueDict):
            self._features = features
        else:
            self._features = TensorValueDict(features)
        self._batch_dims = batch_dims

        self._internals = dict(internals)

    @property
    def batch_dims(self) -> int:
        """The number of batchified dimensions. For the basic State, it should be 0."""
        return self._batch_dims

    @property
    def features(self) -> TensorValueDict:
        return self._features

    @property
    def internals(self) -> Dict[str, Any]:
        """Additional internal information about the state."""
        return self._internals

    def clone(self) -> 'TensorState':
        return type(self)(features=self._features.clone(), batch_dims=self._batch_dims, internals=self.clone_internals())

    def clone_internals(self):
        """Clone the internals."""
        return self.internals.copy()

    def summary_string(self) -> str:
        """Get a summary string of the state. The main difference between this and __str__ is that this function only formats the shape of intermediate tensors."""
        fmt = f'''{type(self).__name__}{{
  states:
'''
        for p in self.features.all_feature_names:
            feature = self.features[p]
            fmt += f'    {p}: {feature.format(content=False)}\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def __str__(self):
        fmt = f'''{type(self).__name__}{{\n'''
        fmt += self.extra_state_str_before()
        fmt += '  states:\n'
        for p in self.features.all_feature_names:
            tensor = self.features[p]
            fmt += f'    - {p}'
            fmt += ': ' + indent_text(str(tensor), level=2).strip() + '\n'
        fmt += self.extra_state_str_after()
        fmt += '}'
        return fmt

    def extra_state_str_before(self) -> str:
        """Extra state string before the features."""
        return ''

    def extra_state_str_after(self) -> str:
        """Extra state string."""
        return ''

ObjectNameArgument = Union[Iterable[str], Mapping[str, ObjectType]]
ObjectTypeArgument = Optional[Iterable[ObjectType]]


class NamedObjectStateMixin(object):
    """A state type mixin with named objects."""

    def __init__(self, object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None):
        """A state type mixin with named objects.
        The object names can be either a list of names, or a mapping from names to :class:`ObjectType`'s.

            - If the `object_names` is a list of names, then the user should also specify a list of object types.
            - If the `object_names` is a mapping from names to :class:`ObjectType`'s, then the `object_types` argument should be None.

        Args:
            object_names: the object names.
            object_types: the object types.
        """
        if isinstance(object_names, Mapping):
            assert object_types is None, 'object_types should be None if object_names is a mapping.'
            self.object_names = tuple(object_names.keys())
            self.object_types = tuple(object_names.values())
        else:
            assert object_types is not None, 'object_types should not be None if object_names is not a mapping.'
            self.object_names = tuple(object_names)
            self.object_types = tuple(object_types)

        self.object_type2name: Dict[str, List[str]] = dict()
        self.object_name2index: Dict[Tuple[str, str], int] = dict()
        self.object_name2defaultindex: Dict[str, Tuple[str, int]] = dict()

        for name, obj_type in zip(self.object_names, self.object_types):
            self.object_type2name.setdefault(obj_type.typename, list()).append(name)
            self.object_name2index[name, obj_type.typename] = len(self.object_type2name[obj_type.typename]) - 1
            self.object_name2defaultindex[name] = obj_type.typename, len(self.object_type2name[obj_type.typename]) - 1
            for t in obj_type.iter_parent_types():
                self.object_type2name.setdefault(t.typename, list()).append(name)
                self.object_name2index[name, t.typename] = len(self.object_type2name[t.typename]) - 1

    @property
    def nr_objects(self) -> int:
        """The number of objects in the current state."""
        return len(self.object_types)

    def get_typed_index(self, name: str, typename: Optional[str] = None) -> int:
        """Get the typed index of the object with the given name.
        There is an additional typename argument to specify the type of the object.
        Because the same object can have multiple types (due to inheritence), the object can have multiple typed indices, one for each type.
        When the typename is not specified, the default type of the object is used (i.e., the most specific type).

        Args:
            name: the name of the object.
            typename: the typename of the object. If not specified, the default type of the object is used (i.e. the most specific type).

        Returns:
            the typed index of the object.
        """
        if typename is None or typename == AutoType.typename:
            return self.object_name2defaultindex[name][1]
        return self.object_name2index[name, typename]

    def get_default_typename(self, name: str) -> str:
        """Get the typename of the object with the given name."""
        return self.object_name2defaultindex[name][0]

    def get_name(self, typename: str, index: int) -> str:
        """Get the name of the object with the given type and index."""
        return self.object_type2name[typename][index]

    def get_objects_by_type(self, typename: str) -> List[str]:
        """Get the names of the objects with the given type."""
        return self.object_type2name[typename]

    def get_nr_objects_by_type(self, typename: str) -> int:
        """Get the number of objects with the given type."""
        return len(self.object_type2name[typename])

    def get_state_object_reference(self, dtype: Union[ObjectType, str], index: Optional[int] = None, name: Optional[str] = None) -> StateObjectReference:
        """Get the object reference with the given type and index."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename

        if index is not None:
            return StateObjectReference(self.get_name(typename, index), index, dtype)
        if name is not None:
            return StateObjectReference(name, self.get_typed_index(name, typename), dtype)
        raise ValueError('Either indices or names should be specified.')

    def get_state_object_list(self, dtype: Union[ObjectType, str], indices: Optional[Union[Sequence[int], slice]] = None, names: Optional[Sequence[str]] = None) -> StateObjectList:
        """Get a list of object references with the given type and indices."""
        if isinstance(dtype, str):
            dtype, typename = ObjectType(dtype), dtype
        else:
            typename = dtype.typename
        if indices is not None:
            if isinstance(indices, slice):
                if indices != QINDEX:
                    raise ValueError('Only QINDEX is allowed for the indices of StateObjectList.')
                return StateObjectList(ListType(dtype), QINDEX)
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, index) for index in indices])
        if names is not None:
            return StateObjectList(ListType(dtype), [self.get_state_object_reference(typename, name=name) for name in names])
        raise ValueError('Either indices or names should be specified.')

class NamedObjectTensorState(TensorState, NamedObjectStateMixin):
    """A state type with named objects."""

    def __init__(self, features: Optional[Union[Mapping[str, Any], MultidimensionalArrayInterface]], object_names: ObjectNameArgument, object_types: ObjectTypeArgument = None, batch_dims: int = 0, internals: Optional[Mapping[str, Any]] = None):
        """Initialize the state.

        Args:
            features: the features of the state.
            object_types: the types of the objects.
            object_names: the names of the objects. If the object_names is a mapping, the object_types should be None.
            batch_dims: the number of batchified dimensions.
            internals: the internals of the state.
        """

        TensorState.__init__(self, features, batch_dims, internals)
        NamedObjectStateMixin.__init__(self, object_names, object_types)

    def clone(self) -> 'NamedObjectTensorState':
        return type(self)(features=self._features.clone(), object_types=self.object_types, object_names=self.object_names, batch_dims=self._batch_dims, internals=self.clone_internals())

    def extra_state_str_before(self) -> str:
        """Extra state string: add the objects."""
        if self.object_names is not None:
            typename2objects = dict()
            for name, dtype in zip(self.object_names, self.object_types):
                typename2objects.setdefault(dtype.typename, list()).append(name)
            objects_str = '; '.join([f'{typename}: [{", ".join(names)}]' for typename, names in typename2objects.items()])
        else:
            objects_str = ', '.join(self.object_names)
        return '  objects: ' + objects_str + '\n'

class QuantizedTensorState:
    def __init__(self, data):
        self.data = data
    
    def quantize(self):
        return self.data

def concat_states(*args: TensorState) -> TensorState:
    """Concatenate a list of states into a batch state.

    Args:
        *args: a list of states.

    Returns:
        a new state, which is the concatenation of the input states.
        This new state will have a new batch dimension.
    """

    if len(args) == 0:raise ValueError('No states to concatenate.')

    all_features = list(args[0].features.all_feature_names)

    # 1. Sanity checks.
    for state in args[1:]:
        assert len(all_features) == len(state.features.all_feature_names)
        for feature in all_features:
            assert feature in state.features.all_feature_names

    # 2. Put the same feature into a list.
    features = {feature_name: list() for feature_name in all_features}
    for state in args:
        for key, value in state.features.tensor_dict.items():
            features[key].append(value)

    # 3. Actually, compute the features.
    feature_names = list(features.keys())
    for feature_name in feature_names:
        features[feature_name] = concat_tvalues(*features[feature_name])

    # 4. Create the new state.
    state = args[0]
    kwargs: Dict[str, Any] = dict()
    if isinstance(state, NamedObjectTensorState):
        kwargs = dict(object_types=state.object_types, object_names=state.object_names)

    kwargs['features'] = features
    kwargs['batch_dims'] = args[0].batch_dims + 1
    return type(state)(**kwargs)

"""action precondition and effect parts,"""

class Precondition:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self, bool_expression) -> bool:
        return 0
    
    def __str__(self): return str(self.bool_expression)

class Effect:
    def __init__(self, bool_expression):
        self.bool_expression = bool_expression
    
    def __call__(self) -> bool:
        return 0
    
    def __str__(self):return str(self.bool_expression)
    
    def split_effect(self):
        """split the effect into two parts. effect+, effect-
        Returns:
            effect+: the added predicates to the known state
            effect-: the removed predicates to the known state
        """
        return 0

class Action:
    def __init__(self, action_name, parameters, precondition, effect, sampler = None):
        """ construct an symboic action with preconditinos and effects
        Args:
            action_name: the name of the action
            precondition: a boolean expression that is callable for an input predicate state
            effect: a set of assignment expressions to known predicates
        """
        super().__init__()
        self.action_name = action_name
        self.parameters = parameters
        if isinstance(precondition, Precondition):
            self.precondition = precondition
        else: self.precondition = Precondition(precondition)

        if not isinstance(effect, Effect):
            self.effect = Effect(effect)
        else: self.effect = effect

        self.sampler  = sampler
    
    def apply(self, state):
        if self.precondition(state):
            return
