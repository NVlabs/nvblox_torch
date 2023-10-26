#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from time import monotonic

import torch


class Timer:
    def __init__(self, noisy=False) -> None:
        self.log = {}
        self.noisy = noisy

    def end(self, name):
        torch.cuda.synchronize()
        now = monotonic()
        assert name in self.log
        self.log[name]["elapsed"] += now - self.log[name]["start"]
        self.log[name]["count"] += 1
        if name != "total":
            self.end("total")
        if self.noisy:
            print(f"End: {name}")

    def start(self, name):
        torch.cuda.synchronize()
        now = monotonic()
        if name not in self.log:
            self.log[name] = {"start": now, "elapsed": 0, "count": 0}
        else:
            self.log[name]["start"] = now
        if name != "total":
            self.start("total")
        if self.noisy:
            print(f"Start: {name}")

    def print(self):
        for name, v in self.log.items():
            avg = v["elapsed"] / v["count"]
            print(f"{name} : Elapsed: {v['elapsed']}, Avg: {avg}")
