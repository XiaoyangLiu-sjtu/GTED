# This code is based on original work by deepseek-ai/DeepSeek-Prover-V1.5.
# The original code is licensed under the MIT License.
# Modifications and adaptations by Authors of OPT, 2025.
# Redistribution and use of this code are also licensed under the MIT License.


import os
import time
import json
import ctypes
import resource
import tempfile
import traceback
import threading
import subprocess
import multiprocessing as mp
import numpy as np
from easydict import EasyDict as AttrDict


HOME_DIR = os.path.expanduser("~")
DEFAULT_LAKE_PATH = f"{HOME_DIR}/.elan/bin/lake"
DEFAULT_LEAN_WORKSPACE = "../ATLAS/src/repl"


class TaskQueue(object):
    def __init__(self, batch_size=512, name="test"):
        self.name = name
        self.batch_size = batch_size
        self.manager = mp.Manager()
        self.waiting_list = self.manager.list()
        self.all_tasks_done = mp.Event()
        self.lock = mp.Lock()
        self._monitor_log = self.manager.list()
        self._monitor_thread = threading.Thread(target=self._monitor)
        self._monitor_thread.start()
    
    def _monitor(self):
        last_log_time = time.time()
        while not self.all_tasks_done.is_set():
            if time.time() - last_log_time >= 60.0:
                with self.lock:
                    if len(self._monitor_log) > 0:
                        print("TaskQueue-{}:  {} requests popped with avg batch_size {:.1f} in last period  {} waiting in queue".format(
                            self.name, np.sum(self._monitor_log), np.mean(self._monitor_log), len(self.waiting_list),
                        ))
                        self._monitor_log[:] = []
                last_log_time = time.time()
            time.sleep(1.0)
    
    def __len__(self):
        return len(self.waiting_list)
    
    def put(self, item):
        with self.lock:
            self.waiting_list.append(item)
    
    def get(self, no_wait=False):
        while not self.all_tasks_done.is_set():
            with self.lock:
                if len(self.waiting_list) > 0:
                    tasks = self.waiting_list[:self.batch_size]
                    self.waiting_list[:self.batch_size] = []
                    self._monitor_log.append(len(tasks))
                    return tasks
            if no_wait:
                break
            time.sleep(0.1)
        return None
    
    def close(self):
        self.all_tasks_done.set()
        self._monitor_thread.join()


class ProcessScheduler(object):
    def __init__(self, batch_size=512, name="test"):
        self.name = name
        self.manager = mp.Manager()
        self.batch_size = batch_size
        self.task_queue = TaskQueue(batch_size=batch_size, name=name)
        self.request_statuses = self.manager.dict()
        self.request_counter = mp.Value(ctypes.c_int32, 0)
        self.lock = mp.Lock()

    def submit_request(self, data):
        with self.lock:
            self.request_counter.value += 1
            request_id = self.request_counter.value
            self.request_statuses[request_id] = None
            self.task_queue.put((time.time(), request_id, data))
        return request_id
    
    def submit_all_request(self, data_list):
        request_id_list = [self.submit_request(data) for data in data_list]
        return request_id_list

    def get_request_status(self, request_id):
        with self.lock:
            response = self.request_statuses.get(request_id, None)
            if response is not None:
                self.request_statuses.pop(request_id)
            return response
    
    def get_request_outputs(self, request_id):
        while True:
            outputs = self.get_request_status(request_id)
            if outputs is not None:
                return outputs
            time.sleep(1.0)
    
    def get_all_request_outputs(self, request_id_list):
        outputs_list = []
        for request_id in request_id_list:
            outputs_list.append(self.get_request_outputs(request_id))
        return outputs_list
    
    def close(self):
        self.task_queue.close()


class Lean4ServerProcess(mp.Process):
    def __init__(self, idx, task_queue, request_statuses, lock, extra_args=AttrDict()):
        super().__init__()
        self.idx = idx
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.extra_args = extra_args
        self.timeout = extra_args.get("timeout", 300)
        self.memory_limit = extra_args.get("memory_limit", -1)
        self.last_output_time = mp.Value(ctypes.c_double, time.time())
        self.complete_count = mp.Value(ctypes.c_int, 0)

    def verify_lean4_file(self, code, lake_path=DEFAULT_LAKE_PATH, lean_workspace=DEFAULT_LEAN_WORKSPACE, last_env=None, verbose=False, timeout=300, allTactics=True, premises=True, tactics=True):
        command = dict(cmd=code, allTactics=allTactics, tactics=tactics, premises=premises)
        if last_env is not None:
            command.update(env=last_env)
        message_str = json.dumps(command, ensure_ascii=False)
        if verbose:
            print(message_str)
        start_time = time.time()
        system_messages = ""
        try:
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
                temp_file.write(message_str + "\r\n\r\n")
                temp_file.seek(0)
                outputs = subprocess.run([lake_path, "exe", "repl"], stdin=temp_file, capture_output=True, text=True, cwd=lean_workspace, timeout=timeout)
            result = json.loads(outputs.stdout)
            result = {
                "sorries" : result.get("sorries", []), 
                "tactics" : result.get("tactics", []),
                "errors" : [m for m in result.get("messages", []) if m["severity"] == "error"],
                "warnings" : [m for m in result.get("messages", []) if m["severity"] == "warning"],
                "infos" : [m for m in result.get("messages", []) if m["severity"] == "info"],
                "system_messages" : system_messages,
                "system_errors" : None,
                "verified_code" : code,
            }
            result["pass"] = not result["errors"]
            result["complete"] = result["pass"] and not result["sorries"] and not any("declaration uses 'sorry'" in warning["data"] or "failed" in warning["data"] for warning in result["warnings"])
        except:
            result = {
                "pass": False,
                "complete": False,
                "system_errors": traceback.format_exc(),
                "system_messages": system_messages
            }
        result["verify_time"] = time.time() - start_time
        return result

    def run(self):
        if self.memory_limit > 0:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (self.memory_limit * (1000 ** 3), self.memory_limit * (1000 ** 3))
            )
        while True:
            inputs = self.task_queue.get()
            if inputs is None: # Terminate when receiving None
                break
            for _, request_id, task in inputs:
                if isinstance(task, str):
                    task = dict(code=task)
                if "timeout" not in task:
                    task["timeout"] = self.timeout
                result = self.verify_lean4_file(**task)
                if len(result["system_messages"]) > 0:
                    retry_start_time = time.time()
                    while ("lean::exception: failed to create thread" in result["system_messages"] or
                           "std::bad_alloc: std::bad_alloc" in result["system_messages"] or
                           "Cannot allocate memory" in result["system_messages"]) \
                          and time.time() - retry_start_time < self.timeout:
                        time.sleep(0.1)
                        result = self.verify_lean4_file(**task)
                with self.lock:
                    self.request_statuses[request_id] = result
                    self.last_output_time.value = time.time()
                    self.complete_count.value += 1


class FLVerifier(ProcessScheduler):
    def __init__(self, max_concurrent_requests=32, timeout=300, memory_limit=-1, name="verifier"):
        super().__init__(batch_size=1, name=name)
        self.processes = [
            Lean4ServerProcess(
                idx=idx,
                task_queue=self.task_queue,
                request_statuses=self.request_statuses,
                lock=self.lock,
                extra_args=AttrDict(
                    timeout=timeout,
                    memory_limit=memory_limit,
                )
            )
            for idx in range(max_concurrent_requests)
        ]
        for p in self.processes:
            p.start()
        print(f"Complete launching {len(self.processes)} LeanServerProcesses")

        self.timeout = timeout
        self._running_monitor = mp.Value(ctypes.c_bool, True)
        self._last_complete_count = mp.Value(ctypes.c_int, 0)
        self._monitor_process = mp.Process(target=self._monitor)
        self._monitor_process.start()
    
    def _monitor(self):
        while self._running_monitor.value:
            time.sleep(1.0)
            subprocess.run(["killall", "repl", f"--older-than={int(self.timeout) + 10}s"], capture_output=True)
    
    def close(self):
        super().close()
        for p in self.processes:
            p.join()
        self._running_monitor.value = False
        self._monitor_process.join()
        print(f"All {len(self.processes)} LeanServerProcesses stopped")