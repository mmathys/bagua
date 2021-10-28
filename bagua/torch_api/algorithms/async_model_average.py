#!/usr/bin/env python3
from bagua.torch_api.bucket import BaguaBucket
from bagua.torch_api.data_parallel import InnerDistributedDataParallel
from bagua.torch_api.algorithms import Algorithm, AlgorithmImpl
from bagua.torch_api.communication import (
    new_group,
    broadcast,
    barrier,
    _pg_group_ranks,
    BaguaProcessGroup,
)
from typing import List
from bagua.torch_api.tensor import BaguaTensor
from bagua.torch_api.env import get_rank
from enum import IntEnum
import threading
import time
import torch
import logging
import concurrent


__all__ = ["AsyncModelAverageAlgorithm", "AsyncModelAverageAlgorithmImpl"]


class _AsyncInternalState(IntEnum):
    RESUME = 0
    ABORT = 1


class AsyncModelAverageAlgorithmImpl(AlgorithmImpl):
    def __init__(
        self,
        process_group: BaguaProcessGroup,
        peer_selection_mode: str = "all",
        sync_interval_ms: int = 500,
        warmup_steps: int = 0,
    ):
        """
        Implementation of the
        `AsyncModelAverage <https://tutorials.baguasys.com/algorithms/async-model-average.html>`_
        algorithm.

        The asynchronous implementation is experimental, and imposes some restrictions.
        With such asynchronous algorithm, the number of iterations on each worker are different. Therefore
        the current implementation assumes that the dataset is an endless stream, and all workers continuously
        synchronize between each other.

        Users should call :meth:`abort` to manually stop the algorithm's continuous synchronization process.
        For example, for a model wrapped with `.with_bagua(...)`, you can abort with `model.bagua_algorithm.abort(model)`,
        and resume with `model.bagua_algorithm.resume(model)`.

        Args:
            process_group (BaguaProcessGroup): The process group to work on.
            peer_selection_mode (str): The way how workers communicate with each other. Currently ``"all"`` is supported.
                ``"all"`` means all workers' weights are synchronized during each communication.
            sync_interval_ms (int): Number of milliseconds between model synchronizations.
            warmup_steps (int): Number of steps to warm up by doing gradient allreduce before doing asynchronous
                model averaging. Use 0 to disable.
        """

        super(AsyncModelAverageAlgorithmImpl, self).__init__(process_group)
        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.step_id = 0
        self.warmup_steps = warmup_steps

        self.cuda_event = torch.cuda.Event()

        self.abort_event = threading.Event()
        self.dummy_tensor = torch.Tensor([0]).byte().cuda()

        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.scheduled = False

        process_ranks = list(_pg_group_ranks[self.process_group])
        self.thread_group = new_group(
            process_ranks, stream=torch.cuda.Stream(priority=-1)
        )

    def tensors_to_buckets(self, tensors: List[List[BaguaTensor]]) -> List[BaguaBucket]:
        if self.step_id < self.warmup_steps:
            return super().tensors_to_buckets(tensors)

        all_tensors = []
        for idx, bucket in enumerate(tensors):
            all_tensors.extend(bucket)

        bagua_bucket = BaguaBucket(all_tensors, flatten=True, name=str(0))

        return [bagua_bucket]

    def init_tensors(self, inner_ddp: InnerDistributedDataParallel) -> List[BaguaTensor]:
        parameters = inner_ddp.bagua_build_params()
        tensors = []
        for name, param in parameters.__reversed__():
            if self.step_id < self.warmup_steps:
                print('name={}, step_id={}, warmup_steps={}'.format(
                    name, self.step_id, self.warmup_steps))
                grad = param.bagua_ensure_grad().ensure_bagua_tensor(
                    name, inner_ddp.bagua_module_name
                )
                param._bagua_grad = grad
                tensors.append(grad)
            else:
                p = param.ensure_bagua_tensor(name, inner_ddp.bagua_module_name)
                tensors.append(p)

        return tensors

    def init_forward_pre_hook(self, inner_ddp: InnerDistributedDataParallel):
        def hook(input):
            if (
                self.step_id > self.warmup_steps
                and self.sync_interval_ms > 0  # noqa: W503
            ):
                self._lock_model(inner_ddp)

                if not hasattr(self, "future"):
                    self.future = self.executor.submit(
                        self._run_async_loop, inner_ddp
                    )
                    self.scheduled = True
                    logging.debug(
                        "Process {} async communication started.".format(get_rank())
                    )

        return hook

    def init_backward_hook(self, inner_ddp: InnerDistributedDataParallel):
        def hook(parameter_name, parameter):
            if self.step_id <= self.warmup_steps:
                parameter._bagua_grad.bagua_mark_communication_ready()

        return hook

    def init_post_backward_hook(self, inner_ddp: InnerDistributedDataParallel):
        def hook():
            if self.step_id <= self.warmup_steps:
                inner_ddp._bagua_backend.wait_pending_comm_ops()
            else:
                self._unlock_model(inner_ddp)

        return hook

    def need_reset(self):
        self.step_id += 1

        if self.warmup_steps > 0 and self.step_id == self.warmup_steps + 1:
            logging.info(f"Async model average starts from step {self.step_id}")
            return True
        else:
            return False

    def init_operations(
        self,
        inner_ddp: InnerDistributedDataParallel,
        bucket: BaguaBucket,
    ):
        inner_ddp._bagua_backend.wait_pending_comm_ops()
        bucket.clear_ops()

        if self.step_id < self.warmup_steps:
            bucket.append_centralized_synchronous_op(
                hierarchical=False,
                average=True,
                group=inner_ddp.process_group,
            )
        else:
            async_op = bucket.append_asynchronous_model_average_op(
                peer_selection_mode=self.peer_selection_mode, group=self.thread_group
            )
            bucket._async_op = async_op

    def _lock_model(self, inner_ddp: InnerDistributedDataParallel):
        torch.cuda.current_stream().record_event(self.cuda_event)
        self.cuda_event.synchronize()

        for bucket in inner_ddp.bagua_buckets:
            bucket._async_op.lock_weight()

    def _unlock_model(self, inner_ddp: InnerDistributedDataParallel):
        torch.cuda.current_stream().record_event(self.cuda_event)
        self.cuda_event.synchronize()

        for bucket in inner_ddp.bagua_buckets:
            bucket._async_op.unlock_weight()

    def _negotiate(self):
        if self.abort_event.is_set():
            self.dummy_tensor[0] = _AsyncInternalState.ABORT
        else:
            self.dummy_tensor[0] = _AsyncInternalState.RESUME

        broadcast(
            self.dummy_tensor,
            src=0,
            comm=self.thread_group.get_global_communicator(),
        )

        return self.dummy_tensor.item()

    def _run_async_loop(self, inner_ddp: InnerDistributedDataParallel):
        comm_step = 0
        while True:
            state = self._negotiate()
            if state == _AsyncInternalState.ABORT:
                break

            start_time = time.time()
            for bucket in inner_ddp.bagua_buckets:
                for tensor in bucket.tensors:
                    tensor.bagua_mark_communication_ready_without_synchronization()

            inner_ddp._bagua_backend.wait_pending_comm_ops()
            duration = (time.time() - start_time) * 1000

            logging.debug(
                "Process {} async communication cost {}ms, comm_step={}".format(
                    get_rank(), duration, comm_step
                )
            )
            comm_step += 1
            time.sleep(self.sync_interval_ms / 1000)

    def abort(self, inner_ddp: InnerDistributedDataParallel):
        """
        Stop background asynchronous communications. Should be called after training.

        Args:
            inner_ddp: A PyTorch module initialized by
                :meth:`~bagua.torch_api.distributed.InnerDistributedDataParallel.with_bagua` method.
        """

        if self.scheduled:
            barrier(comm=inner_ddp.process_group.get_global_communicator())
            self.abort_event.set()
            self.future.result()  # pytype: disable=attribute-error
            self.scheduled = False
            logging.debug("Process {} async communication aborted.".format(get_rank()))

    def resume(self, inner_ddp: InnerDistributedDataParallel):
        """
        Resume aborted background asynchronous communications (see :meth:`abort`). Should be called before training.

        Args:
            inner_ddp: A PyTorch module initialized by
                :meth:`~bagua.torch_api.distributed.InnerDistributedDataParallel.with_bagua` method.
        """

        if not self.scheduled and hasattr(self, "future"):
            barrier(comm=inner_ddp.process_group.get_global_communicator())
            self.abort_event.clear()
            self.future = self.executor.submit(self._run_async_loop, inner_ddp)
            self.scheduled = True
            logging.debug("Process {} async communication resumed.".format(get_rank()))


class AsyncModelAverageAlgorithm(Algorithm):
    def __init__(
        self,
        peer_selection_mode: str = "all",
        sync_interval_ms: int = 500,
        warmup_steps: int = 0,
    ):
        """
        Create an instance of the
        `AsyncModelAverage <https://tutorials.baguasys.com/algorithms/async-model-average.html>`_
        algorithm.

        The asynchronous implementation is experimental, and imposes some restrictions.
        With such asynchronous algorithm, the number of iterations on each worker are different. Therefore
        the current implementation assumes that the dataset is an endless stream, and all workers continuously
        synchronize between each other.

        Users should call :meth:`abort` to manually stop the algorithm's continuous synchronization process.
        For example, for a model wrapped with `.with_bagua(...)`, you can abort with `model.bagua_algorithm.abort(model)`,
        and resume with `model.bagua_algorithm.resume(model)`.

        Args:
            peer_selection_mode (str): The way how workers communicate with each other. Currently ``"all"`` is supported.
                ``"all"`` means all workers' weights are synchronized during each communication.
            sync_interval_ms (int): Number of milliseconds between model synchronizations.
            warmup_steps (int): Number of steps to warm up by doing gradient allreduce before doing asynchronous
                model averaging. Use 0 to disable.
        """

        self.peer_selection_mode = peer_selection_mode
        self.sync_interval_ms = sync_interval_ms
        self.warmup_steps = warmup_steps

    def reify(self, process_group: BaguaProcessGroup) -> AsyncModelAverageAlgorithmImpl:
        return AsyncModelAverageAlgorithmImpl(
            process_group,
            peer_selection_mode=self.peer_selection_mode,
            sync_interval_ms=self.sync_interval_ms,
            warmup_steps=self.warmup_steps,
        )
