#!/usr/bin/env python3

from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from torch.optim.optimizer import Optimizer
import torch
import torch.nn as nn
from csvec import CSVec
from typing import List, Tuple
from bagua.torch_api.tensor import BaguaTensor

DEBUG = False

# Implements SketchSGD encoding and decoding. This is can be used for the stateful
# hook.
class SketchState:
    def __init__(self, optimizer: Optimizer, device=None, c=10, r=10, k=50, momentum=1.0):
        params = optimizer.param_groups[0]["params"]
        
        grad_shape = 0
        for p in params:
            if p.requires_grad:
                grad_shape += p.numel()

        self.grad_shape = grad_shape
        self.optimizer = optimizer
        self.device = device
        self.u = torch.zeros(grad_shape, device=device)
        self.v = torch.zeros(grad_shape, device=device)
        self.momentum = momentum
        self.sketch = CSVec(d=grad_shape, c=c, r=r, device=device)
        self.k = k

    # creates the flattened gradient vector for later encoding in a sketch.
    def _flattened_gradient(self):
        params = self.optimizer.param_groups[0]["params"]
        flattened = []
        for p in params:
            if p.requires_grad:
                assert hasattr(p, "grad"), "gradient must be defined on param (missing backprop?)"
                flattened.append(p.grad.reshape((-1,)))
        res = torch.cat(flattened)
        
        assert len(res) == self.grad_shape, "gradient size mismatch"
        return res 

    # encodes gradient vector into sketch
    def encode(self):
        self.u.mul_(self.momentum)

        gradient = self._flattened_gradient()
        self.u.add_(gradient)

        self.v.add_(self.u)

        self.sketch.zero()
        self.sketch.accumulateVec(self.v)

        return self.sketch.table.clone() 

    # decodes sketch into gradient vector, then applies it to model.
    def decode(self, sketch_table):
        self.sketch.zero()
        self.sketch.table = sketch_table

        # unsketch payload
        gradient = self.sketch.unSketch(k=self.k)
        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0

        # set .grad fields with unsketched gradient vector.
        i = 0
        for p in self.optimizer.param_groups[0]["params"]:
            if p.requires_grad:
                assert hasattr(p, "grad"), "gradient field must exist on param"
                p.grad.set_(gradient[i:i+p.numel()].reshape(p.shape))
                i += p.numel()
        
        assert i == self.grad_shape, "gradient size mismatch"

class SketchAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        optimizer: Optimizer,
        hierarchical: bool = False,
        average: bool = True,
        c=10,
        r=10,
        k=50,
    ):
        super(SketchAlgorithmImpl, self).__init__(process_group)
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.average = average
        self.c = c
        self.r = r
        self.k = k

    def init_tensors(
        self, bagua_ddp: BaguaDistributedDataParallel
    ) -> List[BaguaTensor]:
        parameters = bagua_ddp.bagua_build_params()
        
        tensors = []
        
        name, param = parameters[-1]
        param.sketch = torch.zeros((self.c, self.r), device=param.device)
        param.stepid = 0
        registered_tensor = param.bagua_ensure_grad().ensure_bagua_tensor(
            name,
            bagua_ddp.bagua_module_name,
            getter_closure=lambda param: param.sketch,
            setter_closure=lambda param, t: setattr(param, "sketch", t),
        )
        
        tensors.append(registered_tensor)
        
        self._communication_tensor_names = set((parameters[-1][0],))
        if DEBUG:
            print("----SketchAlgorithmImpl init_tensors batch_idx {} in rank: {}, _communication_tensor_names: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self._communication_tensor_names))
        assert len(self._communication_tensor_names) == len(
            tensors
        ), "tensor names should be unique"
        return tensors

    # Q: what does that do? *it's required*
    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                parameter.bagua_mark_communication_ready()

        return hook

    # Q: what does that do? *it's NOT required*
    def init_post_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook():
            bagua_ddp._bagua_backend.wait_pending_comm_ops()
            self.optimizer.param_groups[0]["params"][-1].stepid += 1

        return hook

    def tensors_to_buckets(
        self, tensors: List[List[BaguaTensor]], do_flatten: bool
    ) -> List[BaguaBucket]:
        if DEBUG:
            print("----SketchAlgorithmImpl tensors_to_buckets batch_idx {} in rank: {}".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device))
        bagua_buckets = []
        for idx, bucket in enumerate(tensors):
            bagua_bucket = BaguaBucket(
                bucket,
                flatten=do_flatten,
                name=str(idx),
                alignment=self.process_group.get_global_communicator().nranks(),
            )
            bagua_buckets.append(bagua_bucket)
        return bagua_buckets

    def init_operations(
        self,
        _: BaguaDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        bucket.clear_ops()

        self.state = None

        def log(*args):
            if not DEBUG: return
            print("----log batch_idx {} in {}: grad---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self.optimizer.param_groups[0]["params"][-1].grad[0:10]))
            for tensor in self.optimizer.param_groups[0]["params"]:
                if tensor.is_bagua_tensor():
                    print("----log batch_idx {} in {}: sketch---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, tensor.sketch))

        def sketch(*args):
            if self.state is None:
                device = bucket.tensors[0].device.type
                self.state = SketchState(self.optimizer, device=device, c=self.c, r=self.r, k = self.k)

            encoded_tensor = self.state.encode()

            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"
            bucket.tensors[0].bagua_setter_closure(encoded_tensor) 

        def unsketch(*args):
            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"

            sketch = bucket.tensors[0].bagua_getter_closure().detach()
            self.state.decode(sketch)

        bucket.append_python_op(log, group=self.process_group)
        bucket.append_python_op(sketch, group=self.process_group)
        bucket.append_python_op(log, group=self.process_group)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )
        bucket.append_python_op(unsketch)

class SketchAlgorithm(Algorithm):
    def __init__(self, optimizer: Optimizer, hierarchical: bool = False, average: bool = True):
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.average = average

    def reify(self, process_group: BaguaProcessGroup) -> SketchAlgorithmImpl:
        return SketchAlgorithmImpl(
            process_group,
            self.optimizer,
            hierarchical=self.hierarchical,
            average=self.average,
        )
