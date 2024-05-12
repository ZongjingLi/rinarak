import itertools

import torch
from torch import nn
from torch.nn import functional as F

from .embedding  import build_box_registry
from .entailment import build_entailment
from .predicates import PredicateFilter
from rinarak.utils import freeze
from rinarak.utils.misc import *
from rinarak.utils.tensor import logit, expat
from rinarak.types import baseType, arrow
from rinarak.program import Primitive, Program

from rinarak.dsl.vqa_types import Boolean

class UnknownArgument(Exception):
    def __init__(self):super()

class UnknownConceptError(Exception):
    def __init__(self):super()

class SceneGraphRepresentation(nn.Module):
    def __init__(self):
        super().__init__()

        self.effective_level = 1
        self.max_level = 4

    @property
    def top_objects(self):
        return 0
    
def get_params(ps, token):
    start_loc = ps.index(token)
    ps = ps[start_loc:]
    count = 0
    outputs = ""
    idx = len(token) + 1
    while count >= 0:
         if ps[idx] == "(": count += 1
         if ps[idx] == ")": count -= 1
         outputs += ps[idx]
         idx += 1
    outputs = outputs[:-1]
    end_loc = idx + start_loc - 1
    return outputs.split(" "), start_loc, end_loc

class CentralExecutor(nn.Module):
    NETWORK_REGISTRY = {}

    def __init__(self, domain, concept_type = "cone", concept_dim = 100):
        super().__init__()
        BIG_NUMBER = 100
        entries = 64

        self.entailment = build_entailment(concept_type, concept_dim)
        self.concept_registry = build_box_registry(concept_type, concept_dim, entries)

        # [Types]
        self.types = domain.types
        for type_name in domain.types:
            baseType(type_name)

        # [Predicate Type Constraints]
        self.type_constraints = domain.type_constraints

        # [Predicates]
        self.predicates = {}
        for predicate in domain.predicates:
            predicate_bind = domain.predicates[predicate]
            predicate_name = predicate_bind["name"]
            params = predicate_bind["parameters"]
            rtype = predicate_bind["type"]

            # check the arity of the predicate
            arity = len(params)

            if arity not in self.predicates:
                self.predicates[arity] = []
            
            #predicate_imp = PredicateFilter(predicate_name,arity)
            self.predicates[arity].append(Primitive(predicate_name,arrow(Boolean, Boolean),predicate_name))
        
        # [Derived]
        self.derived = domain.derived

        # [Actions]
        self.actions = domain.actions

        # [Word Vocab]
        #self.relation_encoder = nn.Linear(config.object_dim * 2, config.object_dim)

        self.concept_vocab = []
        for arity in self.predicates:
            for predicate in self.predicates[arity]:
                self.concept_vocab.append(predicate.name)

        """Neuro Component Implementation Registry"""
        self.implement_registry = {}
        for implement_key in domain.implementations:
            effect = domain.implementations[implement_key]
            self.implement_registry[implement_key] = Primitive(implement_key,arrow(Boolean,Boolean),effect)

        # copy the implementations from the registry

        # args during the execution
        self.kwargs = None 

        self.effective_level = BIG_NUMBER

        self.quantized = False
    
    def check_implementation(self):
        warning = False
        for key in self.implement_registry:
            func_call = self.implement_registry[key]
            if func_call is None:warning = True
        if warning:
            print("Warning: exists predicates not implemented.")
            return False
    
    def redefine_predicate(self, name, func):
        for predicate in Primitive.GLOBALS:
            if predicate== name:
                Primitive.GLOBALS[name].value = func
        return True
 
    def evaluate(self, program, context):
        """program as a string to evaluate under the context
        Args:
            program: a string representing the expression for evaluation
            context: the execution context with predicates and executor
        """
        flat_string = program
        flag = True in [derive in flat_string for derive in self.derived]
        itr = 0
        """Replace all the derived expression in the program with primitives, that means recusion are not allowed"""
        while flag and itr < 9999:
            itr += 1
            for derive_name in self.derived:
                if not "{} ".format(derive_name) in flat_string: continue
                formal_params = self.derived[derive_name]["parameters"]
                actual_params, start, end = get_params(flat_string, derive_name)

                """replace derived expressions with the primtives"""
                prefix = flat_string[:start];suffix = flat_string[end:]
                flat_string = "{}{}{}".format(prefix,self.derived[derive_name]["expr"],suffix)

                for i,p in enumerate(formal_params):flat_string = flat_string.replace(p.split("-")[0], actual_params[i])
            
            """until there are no more derived expression in the program"""
            flag = True in [derive in flat_string for derive in self.derived]
        program = Program.parse(flat_string)
        outputs = program.evaluate(context)
        return outputs
    
    def apply_action(self, action_name, params, context):
        """Apply the action with parameters on the given context
        Args:
            action_name: the name of action to apply
            params: a set of integers represent the index of objects in the scene
            context: given all the observable in a diction
        """
        assert action_name in self.actions
        action = self.actions[action_name] # assert the action must be in the action registry

        """Replace the formal parameters in the predcondition into lambda form"""
        formal_params = [p.split("-")[0] for p in action.parameters]
        context_params = {}
        for i,idx in enumerate(params):
            context_param = {}
            for key in context:
                end = context[key]
                if key != "executor":context_param[key] = end[idx:idx+1]
                else: context_param[key] = self
            context_params[i] = context_param

        # handle the replacements of precondition and effects
        precond_expr = str(action.precondition)
        for i,formal_param in enumerate(formal_params):precond_expr = precond_expr.replace(formal_param, f"${i}")
        effect_expr = str(action.effect)
        for i,formal_param in enumerate(formal_params): effect_expr = effect_expr.replace(formal_param, f"${i}")

        """Evaluate the probabilitstic precondition (not quantized)"""
        precond = self.evaluate(precond_expr,context_params)["end"][0]
        if self.quantized: precond = precond > 0.0 
        else: precond = precond.sigmoid()
        print(precond_expr,precond)
        print(effect_expr)

        """Evaluate the expressions"""
        if self.quantized and not precond: return # if not available, just reduce it
        context_params[0]["end"] = context_params[0]["end"] * (1-precond) + context_params[1]["end"] * precond
        print(context_params[0]["end"])

        # perform value assignment
        effect_output = self.evaluate(effect_expr, context_params)
        #context["is-red"][0] *= (1 - precond)
        #context["is-red"][0] += precond * context["is-red"][2]

    def execute(self,program_result, t = 1.0):
        for result in program_result:
            if "end" in result:
                for cont in result["end"]:
                    self.execute(cont, t)
            if "end1" in result:
                result["end1"] = result["end1"] * (1-t) + result["end2"] * t
        
    def get_implementation(self, func_name):
        func = self.implement_registry[func_name]
        return func
    """Automatically fill up predicates and other symmetry operations"""
    def auto_fillup(self):
        for arity in self.predicates:
            print("arity:",arity)
            for predicate in self.predicates[arity]:
                print(predicate)
        return
    
    def get_type(self, concept):
        concept = str(concept)
        for key in self.type_constraints:
            if concept in self.type_constraints[key]: return key
        return False
    
    def build_relations(self, scene):
        end = scene["end"]
        features = scene["features"]
        N, D = features.shape
        cat_features = torch.cat([expat(features,0,N),expat(features,1,N)], dim = -1)
        relations = self.relation_encoder(cat_features)
        return relations
    
    def spectrum(self,node_features, concepts = None):
        masks = []
        if concepts is None: concepts = self.concept_vocab
        for concept in concepts: 
            masks.append(self.concept_registry(\
                node_features, \
                self.get_concept_embedding(concept))
                )
        return masks

    def entail_prob(self, features, concept):
        kwargs = {"end":[torch.ones(features.shape[0])],
             "features":[features]}
        q = self.parse("filter(scene(),{})".format(concept))
        o = self(q, **kwargs)
        return o["end"]
    
    def all_embeddings(self):
        return self.concept_vocab, [self.get_concept_embedding(emb) for emb in self.concept_vocab]

    def get_concept_embedding(self,concept):
        concept = str(concept)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        concept_index = self.concept_vocab.index(concept)
        idx = torch.tensor(concept_index).unsqueeze(0).to(device)

        return self.concept_registry(idx)

    def forward(self, q, **kwargs):
        self.kwargs = kwargs
        return q(self)
