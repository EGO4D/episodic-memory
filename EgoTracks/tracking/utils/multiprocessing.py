#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

# adopted from PySlowFast

"""Multiprocessing helpers."""

import torch
from detectron2.utils import comm


def run(
    local_rank,
    func,
    num_gpus_per_node,
    init_method,
    machine_rank,
    num_nodes,
    backend,
    args,
    in_node_batch_shuffle=True,
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        machine_rank (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        args: configs. Details in types.py
    """
    # Initialize the process group.
    world_size = num_gpus_per_node * num_nodes
    rank = machine_rank * num_gpus_per_node + local_rank

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    # init distributed in-node group to shuffle only within a node
    # not necessary for now
    assert comm._LOCAL_PROCESS_GROUP is None
    for shard in range(num_nodes):
        node_rank = [num_gpus_per_node * shard + i for i in range(num_gpus_per_node)]
        pg = torch.distributed.new_group(node_rank)
        if shard == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    torch.cuda.set_device(local_rank)
    func(*args)


def launch_job(
    main_func,
    num_gpus_per_node,
    num_machines=1,
    machine_rank=0,
    init_method=None,
    backend="NCCL",
    args=(),
):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (NamedTuple): configs. Details can be found in types.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    torch.multiprocessing.spawn(
        run,
        nprocs=num_gpus_per_node,
        args=(
            main_func,
            num_gpus_per_node,
            init_method,
            machine_rank,
            num_machines,
            backend,
            args,
        ),
    )
