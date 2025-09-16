import random
import numpy as np
from typing import Dict, List, Tuple

import torch
from muxserve.config import MuxServeConfig


def get_gpu_memory(gpu: int = 0) -> int:
    """Returns the total memory of the GPU in bytes."""
    return torch.cuda.get_device_properties(gpu).total_memory


class SMPart:

    def __init__(self, id):
        self.id = id

        self.ref_count = 0

    def acquire(self):
        self.ref_count += 1

    def release(self):
        self.ref_count -= 1

    def is_idle(self):
        return self.ref_count == 0

    def is_overloaded(self):
        return self.ref_count > 1


class SMResource:

    def __init__(self, overload_threshold: int = 4):
        self.overload_threshold = overload_threshold

        self._sm_parts: Dict[int, SMPart] = {}
        self.free_sms: List[SMPart] = []
        self.unoverloaded_sms: List[int] = []
        for i in range(10):
            self._sm_parts[i] = SMPart(i)
            self.free_sms.append(self._sm_parts[i])
            self.unoverloaded_sms.append(i)

    def can_allocate(self, num_sms: int, overload: bool = False) -> bool:
        extra = min(self.overload_threshold, len(
            self.unoverloaded_sms)) if overload else 0
        return len(self.free_sms) + extra >= num_sms

    def allocate(self, num_sms: int, overload: bool = False) -> List[SMPart]:
        ret = []
        for _ in range(num_sms):
            if len(self.free_sms) > 0:
                sm = self.free_sms.pop()
            else:
                sm_id = random.choice(self.unoverloaded_sms)
                sm = self._sm_parts[sm_id]
                self.unoverloaded_sms.remove(sm_id)
            sm.acquire()
            ret.append(sm)
        return ret

    def free(self, sms: List[SMPart]) -> None:
        for sm in sms:
            sm.release()
            if sm.is_idle():
                self.free_sms.append(sm)
            if not sm.is_overloaded() and sm.id not in self.unoverloaded_sms:
                self.unoverloaded_sms.append(sm.id)

    @property
    def num_free_sms(self):
        return len(self.free_sms)

    @property
    def num_overloaded_sms(self):
        return len(self._sm_parts) - len(self.unoverloaded_sms)
