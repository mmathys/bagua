#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
import torch
import torch.nn as nn
from csvec import CSVec


# Implements SketchSGD encoding and decoding. This is can be used for the stateful
# hook.
class SketchState:
    def __init__(self, model=None, grad_shape=None, device=None, c=10, r=10, k=50, momentum=1.0):
        assert (model is None) ^ (grad_shape is None), "either model or grad_shape must be defined"
        
        if model is not None:
            grad_shape = 0
            for p in model.parameters():
                if p.requires_grad:
                    grad_shape += torch.numel(p)

        self.device = device
        self.u = torch.zeros(grad_shape, device=device)
        self.v = torch.zeros(grad_shape, device=device)
        self.momentum = momentum
        self.sketch = CSVec(d=grad_shape, c=c, r=r, device=device)
        self.k = k

    def encode(self, gradient):
        self.u.mul_(self.momentum)
        self.u.add_(gradient)

        self.v.add_(self.u)

        self.sketch.zero()
        self.sketch.accumulateVec(self.v)
        return self.sketch.table.clone() 

    def decode(self, sketch_table):
        self.sketch.zero()
        self.sketch.table = sketch_table
        gradient = self.sketch.unSketch(k=self.k)
        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0

        return gradient

class SketchAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        hierarchical: bool = False,
        average: bool = True,
    ):
        super(SketchAlgorithmImpl, self).__init__(process_group)
        self.hierarchical = hierarchical
        self.average = average

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()

        self.state = None

        # TODO: init and use getter (and setter?) closures for sketch attribute.
        # doublecheck with flattened_tensor property

        def before_op(*args):
            # get shape of bucket
            if self.state is None:
                grad_shape = bucket.flattened_tensor().size()[0]
                device = bucket.tensors[0].device.type
                self.state = SketchState(grad_shape=grad_shape, device=device)

            encoded_tensor = self.state.encode(bucket.flattened_tensor())
            breakpoint()
            bucket.tensors = bucket.tensors[:1]
            bucket.tensors[0].sketch = encoded_tensor

            # Q: bucket.flattened_tensor() still has the same size!

            print(f"before, shape: {len(bucket.tensors)}")
            breakpoint()
            pass

        def after_op(*args):
            print(f"after, shape: {len(bucket.tensors)}")
            assert len(bucket.tensors) == 1, "bucket length should be 1"
            decoded_tensor = self.state.decode(bucket.tensors[0].sketch)
            # Q: how decode original gradients?
            breakpoint()
            pass

        bucket.append_python_op(before_op)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )
        bucket.append_python_op(after_op)


class SketchAlgorithm(Algorithm):
    def __init__(self, hierarchical: bool = False, average: bool = True):
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> SketchAlgorithmImpl:
        return SketchAlgorithmImpl(
            process_group,
            hierarchical=self.hierarchical,
            average=self.average,
        )
