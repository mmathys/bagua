#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel.bagua_distributed import BaguaDistributedDataParallel
from bagua.torch_api.algorithms.base import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import BaguaProcessGroup
from torch.optim.optimizer import Optimizer
import torch
from csvec import CSVec
from typing import List
from bagua.torch_api.tensor import BaguaTensor

DEBUG = False

# Implements SketchSGD encoding and decoding. This is can be used for the stateful
# hook.
class SketchState:
    def __init__(self, optimizer: Optimizer, device=None, c=60, r=5, k=60, momentum=0.0, lr=0.01, sketchParamsLargerThan=0):
        params = optimizer.param_groups[0]["params"]
        for p in params:
            if not hasattr(p, "do_sketching"):
                p.do_sketching = p.numel() >= sketchParamsLargerThan
        
        grad_shape = 0 # size of the gradient
        sketch_shape = 0 # size of the gradient which should be sketched
        sketchMask = [] # controls which parameters should be sketched
        
        for p in params:
            if p.requires_grad:
                size = torch.numel(p)
                grad_shape += size
                if p.do_sketching:
                    sketchMask.append(torch.ones(size))
                    sketch_shape += size
                else:
                    sketchMask.append(torch.zeros(size))

        sketchMask = torch.cat(sketchMask).bool().to(device) 
        assert sketchMask.numel() == grad_shape
        assert sketchMask.sum() == sketch_shape
        self.sketchMask = sketchMask
        self.grad_shape = grad_shape
        self.sketch_shape = sketch_shape

        self.optimizer = optimizer
        
        self.device = device
        self.c = c
        self.r = r
        self.k = k
        self.momentum = momentum
        self.lr = lr
        self.sketch = CSVec(d=sketch_shape, c=c, r=r, device=device)
        self.u = torch.zeros(grad_shape, device=device)
        self.v = torch.zeros(grad_shape, device=device)
        self.logged_bw_savings = False

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

        v_masked = self.v[self.sketchMask]

        self.sketch.zero()
        self.sketch.accumulateVec(v_masked)
        table = self.sketch.table.clone() 

        uncompressed = self.v[~self.sketchMask]
        assert uncompressed.size() == torch.Size([self.grad_shape - self.sketch_shape])

        encoded = torch.cat([table.view(-1), uncompressed])
        
        if not self.logged_bw_savings:
            print(f"Bandwidth savings: from {len(gradient)} to {len(encoded)} ({(len(gradient) / len(encoded)):.2f}x)")
            self.logged_bw_savings = True

        return encoded

    # apply gradient to .grad fields
    def _apply_gradient(self, gradient):
        # set .grad fields with unsketched gradient vector.
        i = 0
        for p in self.optimizer.param_groups[0]["params"]:
            if p.requires_grad:
                if not hasattr(p, "sketch_grad"):
                    p.sketch_grad = torch.zeros_like(p.grad)
                p.grad.set_(gradient[i:i+p.numel()].reshape(p.shape))
                i += p.numel()
        
        assert i == self.grad_shape, "gradient size mismatch"

    # decodes sketch into gradient vector, then applies it to model.
    def decode(self, payload):
        table_len = self.r * self.c
        sketch_table = payload[:table_len].view(self.r, self.c)
        uncompressed = payload[table_len:]

        self.sketch.zero()
        self.sketch.table = sketch_table

        # unsketch payload
        gradient = torch.zeros(self.grad_shape, device=self.device)
        unsketched = self.sketch.unSketch(k=self.k)
        gradient[self.sketchMask] = unsketched

        self.u[gradient.nonzero()] = 0
        self.v[gradient.nonzero()] = 0

        # deal with non-compressed gradients (bias)
        assert uncompressed.size() == torch.Size([self.grad_shape - self.sketch_shape])
        gradient[~self.sketchMask] = uncompressed
        self.v[~self.sketchMask] = 0

        gradient.mul_(self.lr)

        self._apply_gradient(gradient)

class SketchAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        optimizer: Optimizer,
        hierarchical: bool = False,
        average: bool = True,
        c=60,
        r=5,
        k=60,
        lr=0.01,
        momentum=0.0
    ):
        super(SketchAlgorithmImpl, self).__init__(process_group)
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.average = average
        self.c = c
        self.r = r
        self.k = k
        self.lr = lr
        self.momentum = momentum
        device = self.optimizer.param_groups[0]["params"][0].device
        self.state = SketchState(optimizer, device=device, c=c, r=r, k=k, lr=lr, momentum=momentum)

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

    def init_backward_hook(self, bagua_ddp: BaguaDistributedDataParallel):
        def hook(parameter_name, parameter):
            if parameter_name in self._communication_tensor_names:
                parameter.bagua_mark_communication_ready()

        return hook

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

        def log(*args):
            print("----log batch_idx {} in {}: grad---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, self.optimizer.param_groups[0]["params"][-1].grad[0:10]))
            for tensor in self.optimizer.param_groups[0]["params"]:
                if tensor.is_bagua_tensor():
                    print("----log batch_idx {} in {}: sketch---{}.".format(self.optimizer.param_groups[0]["params"][-1].stepid, self.optimizer.param_groups[0]["params"][-1].device, tensor.sketch))

        def sketch(*args):
            encoded_tensor = self.state.encode()

            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"
            bucket.tensors[0].bagua_setter_closure(encoded_tensor) 

        def unsketch(*args):
            assert len(bucket.tensors) == 1, "bucket must only contain a single sketch"
            assert bucket.tensors[0].is_bagua_tensor(), "must be bagua tensor"

            encoded_tensor = bucket.tensors[0].bagua_getter_closure().detach()
            self.state.decode(encoded_tensor)
        
        if DEBUG: bucket.append_python_op(log, group=self.process_group)
        bucket.append_python_op(sketch, group=self.process_group)
        if DEBUG: bucket.append_python_op(log, group=self.process_group)
        bucket.append_centralized_synchronous_op(
            hierarchical=self.hierarchical,
            average=self.average,
            group=self.process_group,
        )
        bucket.append_python_op(unsketch)

class SketchAlgorithm(Algorithm):
    def __init__(self, optimizer: Optimizer, hierarchical: bool = False, average: bool = False, c=60, r=5, k=60, lr=0.01, momentum=0.0):
        self.optimizer = optimizer
        self.hierarchical = hierarchical
        self.average = average
        self.c = c
        self.r = r
        self.k = k
        self.lr = lr
        self.momentum = momentum

    def reify(self, process_group: BaguaProcessGroup) -> SketchAlgorithmImpl:
        return SketchAlgorithmImpl(
            process_group,
            self.optimizer,
            hierarchical=self.hierarchical,
            average=self.average,
            c=self.c,
            r=self.r,
            k=self.k,
            lr=self.lr,
            momentum=self.momentum
        )
