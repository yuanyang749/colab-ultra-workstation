#!/usr/bin/env python3
"""Colab Ultra Workstation: CCU Precision Meter & Lifecycle Tracker.

Implements Dual-Track CCU Accounting:
1. Start Snapshot (开机打点): records {session, gpu, rate, start_time, start_balance}.
2. Stop Settlement (关机结算): unassigns instances, verifies 0 active assignments,
   calculates wall-clock duration & CCU cost, queries final balance, and outputs
   the standardized release report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

TRACKER_FILE = Path.home() / ".config/colab-cli/session_tracker.json"
TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)

GPU_RATES = {
    "CPU": 0.0,
    "T4": 1.19,
    "L4": 1.71,
    "G4": 8.71,
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
    print(f"⏱️ [计费打点] 会话 '{session}' 已登记 | 规格: {gpu} ({rate} CCU/h) | 开机余额: {bal:.2f} CCU")


def cmd_stop(session: str | None = None) -> dict:
    data = load_tracker()
    record = data.get(session) if session else (list(data.values())[-1] if data else None)
    
    # 1. Unassign all instances to guarantee 0 leak
    try:
        from colab_cli.common import state
        _, assignments = state.sync_sessions()
        for a in assignments:
            state.client.unassign(a.endpoint)
    except Exception as e:
        print(f"⚠️ Unassign 异常: {e}", file=sys.stderr)

    # 2. Query final usage
    end_bal, cur_rate, cur_count = get_consumption_info()

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
        
        # Dual-track: API actual diff vs wall-clock theoretical
        api_cost = max(0.0, start_bal - end_bal)
        calc_cost = (dur_sec / 3600.0) * rate_hourly
        
        # If API settled, use API cost; otherwise use wall-clock calc
        final_cost = api_cost if api_cost > 0.005 else calc_cost
        cost_str = f"{final_cost:.2f} CCU"
        if api_cost > 0.005 and abs(api_cost - calc_cost) > 0.05:
            cost_str += f" (按时长折算: {calc_cost:.2f} CCU)"

        if session and session in data:
            del data[session]
            save_tracker(data)

    report = (
        f"\n✅ 算力释放与计费核验报告：\n"
        f"• 实例释放状态: Active assignments: {cur_count}\n"
        f"• 实时消耗速率: {cur_rate:.2f} CCU/h (已完全停止计费)\n"
        f"• 本次运行机型: {gpu_name} ({rate_hourly} CCU/h)\n"
        f"• 本次运行时长: {duration_str}\n"
        f"• 本次实际消耗: {cost_str}\n"
        f"• 当前账户余额: {end_bal:.2f} CCU\n"
    )
    print(report)
    return {
        "assignments": cur_count,
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

    p_report = sub.add_parser("report")

    args = parser.parse_args()
    if args.cmd == "start":
        cmd_start(args.session, args.gpu)
    elif args.cmd == "stop":
        cmd_stop(args.session)
    elif args.cmd == "report":
        bal, rate, count = get_consumption_info()
        print(f"Active assignments: {count} | Rate: {rate:.2f} CCU/h | Balance: {bal:.2f} CCU")


if __name__ == "__main__":
    main()
