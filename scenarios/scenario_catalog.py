from __future__ import annotations

from typing import Any, Dict


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "left_turn": {
        "town": "Town01_Opt",
        "ego_spawn": 224,
        "ego_dest": 83,
        "oncoming_anchor": 173,
        "oncoming_dest": 227,
        "traffic_profile": "normal",
        "mean_headway": 1.00,
        "burst_prob": 0.5,
        "n_oncoming": 30,
    },
    "highway_merge": {
        "town": "Town04_Opt",
        "ego_spawn": 54,
        "ego_dest": 211,
        "oncoming_anchor": 203,
        "oncoming_dest": 137,
        "traffic_profile": "aggressive",
        "mean_headway": 1.00,
        "burst_prob": 0.5,
        "n_oncoming": 30,
    },
}


def apply_scenario_preset(args) -> None:
    scenario_name = getattr(args, "scenario", None)
    if not scenario_name:
        return

    preset = SCENARIO_PRESETS[scenario_name]
    args.town = preset["town"]
    args.ego_spawn = preset["ego_spawn"]
    args.ego_dest = preset["ego_dest"]
    args.oncoming_anchor = preset["oncoming_anchor"]
    args.oncoming_dest = preset["oncoming_dest"]

    if getattr(args, "traffic_profile", None) is None:
        args.traffic_profile = preset["traffic_profile"]
    if getattr(args, "mean_headway", None) is None:
        args.mean_headway = preset["mean_headway"]
    if getattr(args, "burst_prob", None) is None:
        args.burst_prob = preset["burst_prob"]
    if getattr(args, "n_oncoming", None) is None:
        args.n_oncoming = preset["n_oncoming"]
