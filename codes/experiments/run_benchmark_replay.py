#!/usr/bin/env python
# coding: utf-8
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import threading
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
CODES_DIR = THIS_DIR.parent
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from core import Dynamic_ALNS_RL34959
import Dynamic_master34959
from core import Intermodal_ALNS34959
from core import dynamic_RL34959
from core import rl_logging


TRACE_FIELDS = list(dynamic_RL34959.TRACE_FIELDS)
_BASELINE_LOGGER = None


class BaselineLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(self, row_dict):
        row = {k: row_dict.get(k, "") for k in self.fieldnames}
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(row)


def _action_meaning(stage, action_val):
    try:
        action_int = int(action_val)
    except Exception:
        return ""
    if stage and "insert" in stage:
        return "accept_insert" if action_int == 0 else "reject_insert"
    return "wait/keep" if action_int == 0 else "reroute"


def install_baseline_logging(logger):
    def log_event(row_dict, stage, action=None, reward=None, feasible="", source="BASELINE"):
        try:
            if hasattr(row_dict, "to_dict"):
                row_dict = row_dict.to_dict()
        except Exception:
            pass
        if row_dict is None:
            row_dict = {}
        action_val = action if action is not None else row_dict.get("action", "")
        payload = {
            "ts": time.time(),
            "phase": "implement" if getattr(dynamic_RL34959, "implement", 0) == 1 else "train",
            "stage": stage,
            "uncertainty_index": row_dict.get("uncertainty_index", ""),
            "request": row_dict.get("request", ""),
            "vehicle": row_dict.get("vehicle", ""),
            "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
            "dynamic_t_begin": getattr(Intermodal_ALNS34959, "dynamic_t_begin", ""),
            "duration_type": getattr(Intermodal_ALNS34959, "duration_type", ""),
            "gt_mean": getattr(dynamic_RL34959, "current_gt_mean", ""),
            "phase_label": getattr(dynamic_RL34959, "current_phase_label", ""),
            "delay_tolerance": row_dict.get("delay_tolerance", ""),
            "severity": getattr(dynamic_RL34959, "severity_level", ""),
            "passed_terminals": row_dict.get("passed_terminals", ""),
            "current_time": row_dict.get("current_time", ""),
            "action": action_val,
            "reward": reward if reward is not None else row_dict.get("reward", ""),
            "action_meaning": _action_meaning(stage, action_val),
            "feasible": feasible,
            "source": source,
        }
        logger.append(payload)

    dynamic_RL34959.log_trace_from_row = log_event
    Intermodal_ALNS34959.log_rl_event = log_event
    dynamic_RL34959.log_training_row = lambda *args, **kwargs: None
    Intermodal_ALNS34959.log_impl_reward = lambda *args, **kwargs: None
    Intermodal_ALNS34959.save_action_reward_table = lambda *args, **kwargs: None


def initialize_baseline_state():
    dynamic_RL34959.add_event_types = 0
    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 0
    dynamic_RL34959.clear_pairs_done = 1
    dynamic_RL34959.ALNS_got_action_in_implementation = 0
    dynamic_RL34959.evaluate = 0
    dynamic_RL34959.implement = 1
    dynamic_RL34959.wrong_severity_level_with_probability = 0
    dynamic_RL34959.RL_drop_finish = 0
    dynamic_RL34959.add_ALNS = 1
    dynamic_RL34959.state_keys = []
    dynamic_RL34959.time_s = 0
    dynamic_RL34959.episode_length = 1
    dynamic_RL34959.iteration_multiply = 1
    dynamic_RL34959.iteration_numbers_unit = 1
    dynamic_RL34959.total_timesteps2 = 1
    dynamic_RL34959.all_rewards_list = []
    dynamic_RL34959.state_action_reward_collect = dynamic_RL34959.np.array(dynamic_RL34959.np.empty(shape=(0, 9)))
    dynamic_RL34959.state_action_reward_collect_for_evaluate = {}
    dynamic_RL34959.table_number_collect = {}
    dynamic_RL34959.wait_training_finish_last_iteration = 0
    dynamic_RL34959.number_of_state_key = 0
    dynamic_RL34959.state_keys = []
    dynamic_RL34959.iteration_times = 0
    dynamic_RL34959.non_stationary = 0
    Dynamic_ALNS_RL34959.RL_can_start_implementation_phase_from_the_last_table = 1
    Dynamic_ALNS_RL34959.ALNS_calculates_average_duration_list = []
    Dynamic_ALNS_RL34959.reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.removal_reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.removal_state_list_in_implementation = []
    Dynamic_ALNS_RL34959.removal_action_list_in_implementation = []
    Dynamic_ALNS_RL34959.insertion_reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.insertion_state_list_in_implementation = []
    Dynamic_ALNS_RL34959.insertion_action_list_in_implementation = []
    Dynamic_ALNS_RL34959.ALNS_reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.ALNS_removal_reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.ALNS_removal_action_list_in_implementation = []
    Dynamic_ALNS_RL34959.ALNS_insertion_reward_list_in_implementation = []
    Dynamic_ALNS_RL34959.ALNS_insertion_action_list_in_implementation = []
    try:
        Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
    except Exception:
        pass


def emit_reward_event(stage, row_dict, reward, source="BASELINE"):
    try:
        dynamic_RL34959.log_trace_from_row(row_dict, stage, action=row_dict.get("action", ""), reward=reward, source=source)
    except Exception:
        if _BASELINE_LOGGER is None:
            return
        payload = {
            "ts": time.time(),
            "phase": "implement" if getattr(dynamic_RL34959, "implement", 0) == 1 else "train",
            "stage": stage,
            "table_number": getattr(Dynamic_ALNS_RL34959, "table_number", ""),
            "action": row_dict.get("action", ""),
            "reward": reward,
            "source": source,
        }
        _BASELINE_LOGGER.append(payload)


class TrainDriver(threading.Thread):
    def __init__(self, policy_fn, stop_event):
        super().__init__(daemon=True)
        self.policy_fn = policy_fn
        self.stop_event = stop_event
        self.env = dynamic_RL34959.coordinationEnv()

    def run(self):
        while not self.stop_event.is_set():
            try:
                _ = self.env.reset()
            except SystemExit:
                break
            if self.stop_event.is_set():
                break
            action = self.policy_fn()
            try:
                self.env.step(action)
            except SystemExit:
                break


class ImplementDriver(threading.Thread):
    def __init__(self, policy_fn, stop_event):
        super().__init__(daemon=True)
        self.policy_fn = policy_fn
        self.stop_event = stop_event
        self.env = dynamic_RL34959.coordinationEnv()
        self.removal_idx = len(getattr(Dynamic_ALNS_RL34959, "removal_reward_list_in_implementation", []))
        self.insertion_idx = len(getattr(Dynamic_ALNS_RL34959, "insertion_reward_list_in_implementation", []))

    def run(self):
        while not self.stop_event.is_set():
            if not self._wait_for_signal():
                break
            if self.stop_event.is_set():
                break
            try:
                _ = self.env.reset()
            except SystemExit:
                break
            action = self.policy_fn()
            dynamic_RL34959.ALNS_got_action_in_implementation = 0
            dynamic_RL34959.clear_pairs_done = 0
            dynamic_RL34959.send_action(action)
            self._wait_for_accept(action)
            self._flush_reward_lists()

    def _wait_for_signal(self):
        while not self.stop_event.is_set():
            if getattr(Intermodal_ALNS34959, "ALNS_end_flag", 0) == 1:
                return False
            if getattr(Intermodal_ALNS34959, "ALNS_implement_start_RL_can_move", 0) == 1:
                Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
                return True
            time.sleep(0.01)
        return False

    def _wait_for_accept(self, action):
        while not self.stop_event.is_set():
            if getattr(Intermodal_ALNS34959, "ALNS_end_flag", 0) == 1:
                return
            try:
                if len(Intermodal_ALNS34959.state_reward_pairs) > 0:
                    if Intermodal_ALNS34959.state_reward_pairs.iloc[0]["action"] == -10000000:
                        dynamic_RL34959.send_action(action)
            except Exception:
                pass
            try:
                pairs_len = len(Intermodal_ALNS34959.state_reward_pairs)
            except Exception:
                pairs_len = 0
            if dynamic_RL34959.ALNS_got_action_in_implementation == 1 or pairs_len == 0:
                try:
                    Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
                except Exception:
                    pass
                Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
                dynamic_RL34959.ALNS_got_action_in_implementation = 0
                dynamic_RL34959.clear_pairs_done = 1
                return
            time.sleep(0.01)

    def _flush_reward_lists(self):
        removal_rewards = getattr(Dynamic_ALNS_RL34959, "removal_reward_list_in_implementation", [])
        removal_states = getattr(Dynamic_ALNS_RL34959, "removal_state_list_in_implementation", [])
        removal_actions = getattr(Dynamic_ALNS_RL34959, "removal_action_list_in_implementation", [])
        while self.removal_idx < len(removal_rewards):
            reward = removal_rewards[self.removal_idx]
            state_row = removal_states[self.removal_idx] if self.removal_idx < len(removal_states) else {}
            if hasattr(state_row, "to_dict"):
                state_row = state_row.to_dict()
            if not isinstance(state_row, dict):
                state_row = {}
            action_val = removal_actions[self.removal_idx] if self.removal_idx < len(removal_actions) else state_row.get("action", "")
            state_row["action"] = action_val
            emit_reward_event("finish_removal", state_row, reward)
            self.removal_idx += 1

        insertion_rewards = getattr(Dynamic_ALNS_RL34959, "insertion_reward_list_in_implementation", [])
        insertion_states = getattr(Dynamic_ALNS_RL34959, "insertion_state_list_in_implementation", [])
        insertion_actions = getattr(Dynamic_ALNS_RL34959, "insertion_action_list_in_implementation", [])
        while self.insertion_idx < len(insertion_rewards):
            reward = insertion_rewards[self.insertion_idx]
            state_row = insertion_states[self.insertion_idx] if self.insertion_idx < len(insertion_states) else {}
            if hasattr(state_row, "to_dict"):
                state_row = state_row.to_dict()
            if not isinstance(state_row, dict):
                state_row = {}
            action_val = insertion_actions[self.insertion_idx] if self.insertion_idx < len(insertion_actions) else state_row.get("action", "")
            state_row["action"] = action_val
            emit_reward_event("finish_insertion", state_row, reward)
            self.insertion_idx += 1



def load_table_phase_sequence(trace_path):
    sequence = []
    prev = None
    with trace_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("table_number", "")
            if raw is None or raw == "":
                continue
            try:
                value = float(raw)
            except Exception:
                continue
            if math.isnan(value):
                continue
            table_number = int(value)
            phase = str(row.get("phase", "") or "").strip().lower()
            if phase not in {"train", "implement"}:
                phase = "train"
            if prev is None or table_number != prev:
                sequence.append((table_number, phase))
                prev = table_number
    return sequence


def load_request_number(run_dir):
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "request_number" in payload:
                return int(payload["request_number"])
        except Exception:
            pass
    match = re.search(r"_R(\d+)", run_dir.name)
    if match:
        return int(match.group(1))
    return 5


def build_policy(policy_name, seed):
    if policy_name == "wait":
        return lambda: 0
    if policy_name == "reroute":
        return lambda: 1
    if policy_name == "random":
        rng = random.Random(seed)
        return lambda: rng.choice([0, 1])
    raise ValueError(f"unknown policy: {policy_name}")


def apply_phase(phase):
    is_impl = phase == "implement"
    dynamic_RL34959.implement = 1 if is_impl else 0
    dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 0
    dynamic_RL34959.wait_training_finish_last_iteration = 0
    return is_impl


def run_policy(run_dir, table_sequence, request_number, policy_name, seed):
    global _BASELINE_LOGGER
    baseline_path = run_dir / f"baseline_{policy_name}.csv"
    logger = BaselineLogger(baseline_path, TRACE_FIELDS)
    install_baseline_logging(logger)
    _BASELINE_LOGGER = logger

    initialize_baseline_state()
    Dynamic_master34959.add_RL = 1
    os.environ["DYNAMIC_DATA_ROOT"] = str(run_dir / "data")
    os.environ["ALNS_OUTPUT_ROOT"] = str(run_dir)
    Intermodal_ALNS34959.refresh_figures_dir()
    rl_logging.set_run_dir(str(run_dir))

    policy_fn = build_policy(policy_name, seed)

    for table_number, phase in table_sequence:
        is_impl = apply_phase(phase)
        stop_event = threading.Event()
        driver = ImplementDriver(policy_fn, stop_event) if is_impl else TrainDriver(policy_fn, stop_event)
        driver.start()

        try:
            Dynamic_ALNS_RL34959.table_number = int(table_number)
            dynamic_RL34959.clear_pairs_done = 1
            Intermodal_ALNS34959.ALNS_implement_start_RL_can_move = 0
            try:
                Intermodal_ALNS34959.state_reward_pairs = Intermodal_ALNS34959.state_reward_pairs.iloc[0:0]
            except Exception:
                pass
            try:
                Intermodal_ALNS34959.ALNS_end_flag = 0
            except Exception:
                pass
            Dynamic_ALNS_RL34959.Intermodal_ALNS_function(request_number)
        except SystemExit:
            pass
        finally:
            stop_event.set()
            if not is_impl:
                dynamic_RL34959.stop_everything_in_learning_and_go_to_implementation_phase = 1
            driver.join(timeout=5)
            if is_impl:
                try:
                    driver._flush_reward_lists()
                except Exception:
                    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="path to run_YYYYMMDD... folder")
    parser.add_argument("--policy", default="all", choices=["wait", "reroute", "random", "all"])
    parser.add_argument("--include-random", action="store_true", help="include random when policy=all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tables", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    trace_path = run_dir / "rl_trace.csv"
    if not trace_path.exists():
        raise SystemExit(f"missing rl_trace.csv in {run_dir}")

    table_sequence = load_table_phase_sequence(trace_path)
    if args.max_tables > 0:
        table_sequence = table_sequence[: args.max_tables]
    if not table_sequence:
        raise SystemExit("table sequence is empty")

    request_number = load_request_number(run_dir)

    os.environ["STOP_FLAG_FILE"] = str(run_dir / "34959.txt")
    stop_flag = os.environ.get("STOP_FLAG_FILE", "34959.txt")
    if os.path.exists(stop_flag):
        try:
            os.remove(stop_flag)
        except Exception:
            pass

    policies = [args.policy]
    if args.policy == "all":
        policies = ["wait", "reroute"]
        if args.include_random:
            policies.append("random")

    for policy in policies:
        run_policy(run_dir, table_sequence, request_number, policy, args.seed)


if __name__ == "__main__":
    main()
