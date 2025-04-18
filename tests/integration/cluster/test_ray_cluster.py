# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Any, Dict, Generator

import pytest

try:
    import ray  # type: ignore[import-not-found]
    from ray.util.scheduling_strategies import (  # type: ignore[import-not-found]
        PlacementGroupSchedulingStrategy,
    )

    _has_ray = True
except ImportError:
    _has_ray = False


from fairseq2.cluster import RayClusterHandler, RayCoordinator

if _has_ray:

    @pytest.fixture(scope="module")
    def ray_cluster() -> Generator[Any, Any, Any]:
        """Start and stop Ray for the duration of these tests."""
        if not ray.is_initialized():
            ray.init(num_cpus=4)  # Adjust based on your local machine

        yield

        if ray.is_initialized():
            ray.shutdown()

    @pytest.fixture
    def cluster_setup() -> Generator[Any, Any, Any]:
        """Set up test configuration and placement groups."""
        # Test configuration
        num_nodes = 2
        cpus_per_node = 2
        job_id = random.randint(1, 1000000)

        # Create placement groups (simulating nodes)
        placement_groups = []
        for i in range(num_nodes):
            pg = ray.util.placement_group(
                bundles=[{"CPU": 1} for _ in range(cpus_per_node)],
                strategy="STRICT_PACK",
            )
            ray.get(pg.ready())
            placement_groups.append(pg)

        # Create coordinator
        coordinator_name = f"coordinator_{job_id}"
        RayCoordinatorActor = ray.remote(RayCoordinator)
        coordinator = RayCoordinatorActor.options(  # type: ignore[attr-defined]
            name=coordinator_name,
            namespace="fairseq2",
            num_cpus=0,
        ).remote(
            job_id=job_id,
            world_size=num_nodes * cpus_per_node,  # Using CPUs instead of GPUs
        )
        ray.get(coordinator.get_leader_info.remote())

        # Return test configuration
        setup = {
            "num_nodes": num_nodes,
            "cpus_per_node": cpus_per_node,
            "job_id": job_id,
            "placement_groups": placement_groups,
            "coordinator_name": coordinator_name,
        }

        yield setup

    def test_ray_cluster_coordination(
        ray_cluster: None, cluster_setup: Dict[str, Any]
    ) -> None:
        """Test that the RayClusterHandler instances correctly coordinate through RayCoordinator."""
        num_nodes = cluster_setup["num_nodes"]
        cpus_per_node = cluster_setup["cpus_per_node"]
        placement_groups = cluster_setup["placement_groups"]
        coordinator_name = cluster_setup["coordinator_name"]

        # Create and test workers
        @ray.remote
        class TestWorker:
            def __init__(
                self, rank: int, local_rank: int, local_world_size: int
            ) -> None:
                self.env = {
                    "RAY_FAIRSEQ2_COORDINATOR_NAME": f"{coordinator_name}:fairseq2",
                    "RANK": str(rank),
                    "LOCAL_RANK": str(local_rank),
                    "LOCAL_WORLD_SIZE": str(local_world_size),
                    "CUDA_VISIBLE_DEVICES": str(
                        bundle_idx
                    ),  # Simulate GPU ID with CPU ID
                }
                self.cluster_handler = RayClusterHandler(self.env)

            def run_test(self) -> Dict[str, Any]:
                self.cluster_handler.set_torch_distributed_variables()

                # Return environment after setup
                return self.env

        # Create all workers
        workers = []
        for pg_idx in range(num_nodes):
            for bundle_idx in range(cpus_per_node):
                # Place the worker in the appropriate placement group
                worker = TestWorker.options(  # type: ignore[attr-defined]
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_groups[pg_idx],
                        placement_group_bundle_index=bundle_idx,
                    )
                ).remote(
                    rank=pg_idx * cpus_per_node + bundle_idx,
                    local_rank=bundle_idx,
                    local_world_size=cpus_per_node,
                )
                workers.append(worker)

        # Run all workers
        results = ray.get([worker.run_test.remote() for worker in workers])

        # Check results
        assert len(results) == num_nodes * cpus_per_node

        # All workers should have the same WORLD_SIZE
        expected_world_size = num_nodes * cpus_per_node
        for env in results:
            assert int(env["WORLD_SIZE"]) == expected_world_size

        # Check that ranks are assigned correctly (0 to total_workers-1)
        ranks = [int(env["RANK"]) for env in results]
        assert sorted(ranks) == list(range(expected_world_size))

        # All workers should agree on the same master
        master_addr = results[0]["MASTER_ADDR"]
        master_port = results[0]["MASTER_PORT"]
        for env in results:
            assert env["MASTER_ADDR"] == master_addr
            assert env["MASTER_PORT"] == master_port
