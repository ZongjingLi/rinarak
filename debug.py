from rinarak.program import *
from rinarak.types import *

tone = Primitive("1",arrow(tint), 1)
plus = Primitive("plus",arrow(tint,tint,tint), lambda x: lambda y: x + y)

p = Program.parse("(plus 1 1)")

o = p.evaluate({})

print(o)