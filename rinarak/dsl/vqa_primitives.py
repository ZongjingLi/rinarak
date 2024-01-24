from .vqa_types import *
# [Exist at Least one Element in the Set]
t_exists = Primitive(
    "exists",
    arrow(ObjectSet, Boolean),
    lambda x:{"end":torch.max(x["end"]), "executor":x["executor"]})

# [Filter Attribute Concept]
def type_filter(objset,concept,executor):
    filter_logits = torch.zeros_like(objset["end"])
    parent_type = executor.get_type(concept)
    for candidate in executor.type_constraints[parent_type]:
        filter_logits += executor.entailment(objset["features"],
            executor.get_concept_embedding(candidate)).sigmoid()

    div = executor.entailment(objset["features"],
            executor.get_concept_embedding(concept)).sigmoid()
    filter_logits = logit(div / filter_logits)
    return{"end":torch.min(objset["end"],filter_logits), "executor":objset["executor"]}
    
"""This method will be deprecated soon, please use eFilter operator"""
tFilter = Primitive(
    "filter",
    arrow(ObjectSet, Concept, ObjectSet),
    lambda objset: lambda concept: type_filter(objset, concept, objset["executor"]))

def expression_filter(objset, expr, executor):
    expr_logits = executor.evaluate(expr, objset["features"])
    return {"end":torch.min(objset["end"], expr_logits)}

eFilter = Primitive(
    "filter_expr()",
    arrow(ObjectSet, BooleanExpression, ObjectSet),
    lambda objset: lambda expr: expression_filter(objset, expr, objset["executor"]))

def relate(x,y,z):
    EPS = 1e-6;
    #expand_maks = torch.matmul
    mask = x["executor"].entailment(x["relations"],x["executor"].get_concept_embedding(z))
    N, N = mask.shape
 
    score_expand_mask = torch.min(expat(x["end"],0,N),expat(x["end"],1,N))

    new_logits = torch.min(mask, score_expand_mask)

    return {"end":new_logits, "executor":x["executor"]}
def Relate(x):
    return lambda y: lambda z: relate(x,y,z)
tRelate = Primitive("relate",arrow(ObjectSet, ObjectSet, Concept, ObjectSet), Relate)

# [Intersect Sets]{
def Intersect(x): return lambda y: {"end":torch.min(x, y)}
tIntersect = Primitive("intersect",arrow(ObjectSet, ObjectSet, ObjectSet), Intersect)

def Union(x): return lambda y: {"end":torch.max(x["end"], y["end"])}
tUnion = Primitive("equal",arrow(ObjectSet, ObjectSet, ObjectSet), Union)

# [Do Some Counting]
def Count(x):return {"end":torch.sigmoid(x["end"]).sum(-1), "executor":x["executor"]}
tCount = Primitive("count",arrow(ObjectSet, tint), Count)

def Equal(x):return lambda y:  {"end":8 * (.5 - (x - y).abs()), "executor":x["executor"]}
tEqual = Primitive("equal",arrow(treal, treal, Boolean), Equal)

def If(x,y): x["executor"].execute(y,x["end"])

tIf = Primitive("if", arrow(Boolean, CodeBlock), If)

def Assign(x, y):return {"end1":x["end"], "end2":y["end"]}
tAssign = Primitive("assign", arrow(Attribute, Attribute), Assign)

def Forall(condition,set): return 
tForall = Primitive("forall", Boolean, Forall)

def And(x): return lambda y: {"end":torch.min(x["end"],y["end"])}
tAnd = Primitive("and", arrow(Boolean, Boolean, Boolean), And)

def Or(x): return lambda y: {"end": torch.max(x["end"],y["end"])}
tOr = Primitive("or", arrow(Boolean, Boolean, Boolean), Or)

tTrue = Primitive("true",Boolean,{"end":logit(torch.tensor(1.0))})
tFalse = Primitive("false",Boolean,{"end":logit(torch.tensor(0.0))})

tPr = Primitive("Pr", arrow(ObjectSet,Concept,ObjectSet),
    lambda x: lambda y: {"logits":x["executor"].entailment(x["features"],
            x["executor"].get_concept_embedding(y))}
)