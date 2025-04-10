
# -*- coding: utf-8 -*-
# @Author: zongjingli
# @Date:   2025-02-06 06:22:49
# @Last Modified by:   zongjingli
# @Last Modified time: 2025-02-19 20:23:27

import open3d as o3d
from helchriss.domain import load_domain_string, Domain
from helchriss.knowledge.executor import CentralExecutor

blockworld_domain_str = """
(domain Blockworld)
(:type
    state - vector[float,3] ;; encoding of position and is holding
    position - vector[float,2]
    scene - List[ vector[float,2] ]
)
(:predicate
    block_position ?x-state -> position
    on ?x-state ?y-state -> boolean
    clear-above ?x-state -> boolean
    holding ?x-state -> boolean
    hand_free -> boolean
)

(:action
    (
        name: pick
        parameters: ?o1
        precondition: (and (clear ?o1) (hand-free) )
        effect:
        (and-do
            (and-do
                (assign (holding ?o1) true)
                (assign (clear ?o1) false)
            )
            (assign (hand-free) false)
        )
    )
    (
        name: place
        parameters: ?o1 ?o2
        precondition:
            (and (holding ?o1) (clear ?o2))
        effect :
            (and-do
            (and-do
                        (assign (hand-free) true)
                (and-do
                        (assign (holding ?o1) false)
                    (and-do
                        (assign (clear ?o2) false)
                        (assign (clear ?o1) true)
                    )
                )
                
            )
                (assign (on ?x ?y) true)
            )
    )
)
"""
import torch
import torch.nn as nn

class BlockworldExecutor(CentralExecutor):

    def __init__(self, domain, concept_dim):
        super().__init__(domain, concept_dim)
        self.neural_part = nn.Linear(2, 2)

    def block_position(self, i):
        return self.neural_part(self.grounding["block_position"][i])
    
    def holding(self):
        return self.grounding
    
    def hand_free(self):
        return self.grounding["hand_free"]
    
    def on(self, i,j):
        return self.grounding["block_position"][i] - self.grounding["block_position"][j]

blockworld_domain = load_domain_string(blockworld_domain_str)
blockworld_domain.print_summary()

blockworld_executor = BlockworldExecutor(blockworld_domain, concept_dim = 128)


if __name__ == "__main__":
    import torch

    positions = torch.randn([4,2])
    positions.requires_grad = True
    context = {"block_position" : positions}
    context["hand_free"] = True

    res = blockworld_executor.evaluate("hand_free()", context)
    print(res)

    res = blockworld_executor.evaluate("block_position(2)", context)
    print(res)

    res = blockworld_executor.evaluate("on(1,2)", context)

    print(res)

    from helchriss.knowledge.symbolic import Expression
    expr = Expression.parse_program_string("point:Path(1, 2)")
    print(expr)

    optimizer = torch.optim.Adam(blockworld_executor.parameters(), lr = 1e-2)
    for epoch in range(500):
        res = blockworld_executor.evaluate("block_position(2)", context).value

        loss = torch.nn.functional.mse_loss(res, torch.tensor([1.0, 0.5]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


