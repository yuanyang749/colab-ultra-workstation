#!/usr/bin/env python3
"""Colab Ultra Workstation: CCU Precision Meter & Lifecycle Tracker.

Implements Dual-Track CCU Accounting (Multi-Session Safe):
1. Start Snapshot (开机打点): records {session, gpu, rate, start_time, start_balance}.
2. Stop Settlement (单会话精准关机结算): unassigns ONLY the specified session (-s <session>),
   lists any remaining concurrent sessions without killing them, calculates wall-clock
   duration & CCU cost for the current session, queries final balance, and outputs
   the standardized release report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

TRACKER_FILE = Path.home() / ".config/colab-cli/session_tracker.json"
TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)

GPU_RATES = {
    "CPU": 0.0,
    "T4": 1.19,
    "L4": 1.71,
    "G4": 10.05,
    "A100": 8.71,
    "H100": 18.05,
}


def get_consumption_info() -> tuple[float, float, int]:
    """Fetch current balance, hourly rate, and active assignments count via colab_cli."""
    try:
        from colab_cli.common import state
        info = state.client.get_consumption_user_info()
        return (
            float(info.paid_compute_units_balance),
            float(info.consumption_rate_hourly),
            int(info.assignments_count),
        )
    except Exception as e:
        print(f"⚠️ 无法获取 CCU 余额: {e}", file=sys.stderr)
        return (0.0, 0.0, 0)


def get_active_sessions_summary() -> list[str]:
    """Return human-readable summary strings for all currently active Colab sessions."""
    items: list[str] = []
    try:
        from colab_cli.common import state
        sessions, assignments = state.sync_sessions()
        tracked_endpoints = set()
        for sid, s in sessions.items():
            ep = getattr(s, "endpoint", None)
            hw = getattr(s, "accelerator", None) or getattr(s, "variant", "CPU")
            if ep:
                tracked_endpoints.add(ep)
                items.append(f"[{sid}] {hw} ({ep})")
        for a in assignments:
            ep = getattr(a, "endpoint", None)
            if ep and ep not in tracked_endpoints:
                hw = getattr(a, "accelerator", None) or getattr(a, "variant", "GPU")
                items.append(f"[untracked] {hw} ({ep})")
    except Exception:
        pass
    return items


def load_tracker() -> dict:
    if TRACKER_FILE.exists():
        try:
            return json.loads(TRACKER_FILE.read_text())
        except Exception:
            pass
    return {}


def save_tracker(data: dict) -> None:
    TRACKER_FILE.write_text(json.dumps(data, indent=2))


def cmd_start(session: str, gpu: str) -> None:
    gpu = gpu.upper()
    rate = GPU_RATES.get(gpu, 0.0)
    bal, cur_rate, count = get_consumption_info()
    data = load_tracker()
    data[session] = {
        "session": session,
        "gpu": gpu,
        "rate": rate,
        "start_time": time.time(),
        "start_balance": bal,
    }
    save_tracker(data)
    print(
        f"⏱️ [计费打点] 会话 '{session}' 已登记 | 规格: {gpu} ({rate} CCU/h) "
        f"| 当前在线实例: {count} | 开机余额: {bal:.2f} CCU"
    )


def cmd_stop(session: str | None = None, stop_all: bool = False) -> dict:
    data = load_tracker()
    if not session and not stop_all and data:
        # Default to the most recently started tracked session rather than killing all
        session = list(data.keys())[-1]
    record = data.get(session) if session else None

    # 1. Unassign ONLY the target session (unless --all is explicitly passed)
    try:
        from colab_cli.common import state
        if stop_all:
            _, assignments = state.sync_sessions()
            for a in assignments:
                state.client.unassign(a.endpoint)
            for sid in list(state.store.list_all().keys()):
                state.store.remove(sid)
        elif session:
            state.sync_sessions()
            s = state.store.get(session)
            if s and getattr(s, "endpoint", None):
                state.client.unassign(s.endpoint)
            state.store.remove(session)
    except Exception as e:
        print(f"⚠️ Unassign 异常: {e}", file=sys.stderr)

    # 2. Query final usage and remaining sessions
    end_bal, cur_rate, cur_count = get_consumption_info()
    remaining_sessions = get_active_sessions_summary()

    duration_str = "未知"
    cost_str = "0.00 CCU"
    gpu_name = "GPU"
    rate_hourly = 0.0

    if record:
        start_t = record.get("start_time", time.time())
        start_bal = record.get("start_balance", end_bal)
        gpu_name = record.get("gpu", "GPU")
        rate_hourly = record.get("rate", 0.0)
        dur_sec = max(0, time.time() - start_t)

        mins = int(dur_sec // 60)
        secs = int(dur_sec % 60)
        duration_str = f"{mins} 分 {secs} 秒"

        api_cost = max(0.0, start_bal - end_bal)
        calc_cost = (dur_sec / 3600.0) * rate_hourly

        final_cost = api_cost if api_cost > 0.005 else calc_cost
        cost_str = f"{final_cost:.2f} CCU"
        if api_cost > 0.005 and abs(api_cost - calc_cost) > 0.05:
            cost_str += f" (按当前会话时长折算: {calc_cost:.2f} CCU)"

        if session and session in data:
            del data[session]
            save_tracker(data)

    target_label = f"会话 '{session}' 已释放" if session else "全部会话已释放"
    rate_status = "已完全停止计费" if cur_count == 0 else f"当前会话已停止计费，另有 {cur_count} 个并行实例运行中"
    remaining_line = (
        f"• 其他在线会话: {', '.join(remaining_sessions)}\n"
        if remaining_sessions
        else ("• 其他在线会话: 无 (当前无其他并行实例)\n")
    )

    report = (
        f"\n✅ 算力释放与计费核验报告：\n"
        f"• 实例释放状态: {target_label} | Active assignments: {cur_count}\n"
        f"{remaining_line}"
        f"• 实时消耗速率: {cur_rate:.2f} CCU/h ({rate_status})\n"
        f"• 本次运行机型: {gpu_name} ({rate_hourly} CCU/h)\n"
        f"• 本次运行时长: {duration_str}\n"
        f"• 本次实际消耗: {cost_str}\n"
        f"• 当前账户余额: {end_bal:.2f} CCU\n"
    )
    print(report)
    return {
        "assignments": cur_count,
        "remaining_sessions": remaining_sessions,
        "rate": cur_rate,
        "duration": duration_str,
        "cost": cost_str,
        "balance": end_bal,
    }


def main():
    parser = argparse.ArgumentParser(description="Colab CCU Meter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start")
    p_start.add_argument("-s", "--session", required=True)
    p_start.add_argument("-g", "--gpu", default="CPU")

    p_stop = sub.add_parser("stop")
    p_stop.add_argument("-s", "--session", default=None)
    p_stop.add_argument("--all", action="store_true", help="Force unassign all active sessions")

    p_report = sub.add_parser("report")

    args = parser.parse_args()
    if args.cmd == "start":
        cmd_start(args.session, args.gpu)
    elif args.cmd == "stop":
        cmd_stop(args.session, stop_all=args.all)
    elif args.cmd == "report":
        bal, rate, count = get_consumption_info()
        rem = get_active_sessions_summary()
        rem_str = f" | Sessions: {', '.join(rem)}" if rem else ""
        print(f"Active assignments: {count} | Rate: {rate:.2f} CCU/h | Balance: {bal:.2f} CCU{rem_str}")


if __name__ == "__main__":
    main()
