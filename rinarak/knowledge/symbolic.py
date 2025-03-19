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


from termcolor import colored
from dataclasses import dataclass
from typing import Optional
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
    def __init__(self, data):
        """ construct a symbolic or hybrid state
        Args:
            data: a diction that maps the data of each state to the actual value
        """
        self.data = data

""""""

class TensorState(State):
    def __init__(self, data):
        super(data)


class QuantizedTensorState:
    def __init__(self, data):
        self.data = data
    
    def quantize(self):
        return self.data

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
