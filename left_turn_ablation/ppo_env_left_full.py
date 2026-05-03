#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import carla

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(THIS_DIR, "scenarios"))

if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
if SOURCE_DIR not in sys.path:
    sys.path.insert(0, SOURCE_DIR)

from scene import build_scene, ensure_agents_importable, reset_vehicle
from scenario_catalog import SCENARIO_PRESETS

ensure_agents_importable()
from agents.navigation.controller import VehiclePIDController  # type: ignore
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore


def _xy(loc: carla.Location) -> np.ndarray:
    return np.array([float(loc.x), float(loc.y)], dtype=np.float32)


def _vec_mag(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _speed_kmh(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6 * _vec_mag(vel.x, vel.y, vel.z)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _dedupe_polyline(pts: List[np.ndarray], tol: float = 0.5) -> List[np.ndarray]:
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > tol:
            out.append(p)
    return out


def _distance_point_to_polyline(pt: np.ndarray, poly: List[np.ndarray]) -> float:
    if len(poly) == 0:
        return 1e9
    if len(poly) == 1:
        return float(np.linalg.norm(pt - poly[0]))
    best = 1e9
    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-6:
            d = float(np.linalg.norm(pt - a))
        else:
            u = float(np.dot(pt - a, ab) / denom)
            u = max(0.0, min(1.0, u))
            proj = a + u * ab
            d = float(np.linalg.norm(pt - proj))
        best = min(best, d)
    return best


def _extract_junction_route_xy(route_waypoints: List[carla.Waypoint]) -> List[np.ndarray]:
    pts: List[np.ndarray] = []
    started = False
    for wp in route_waypoints:
        if wp.is_junction and not started:
            started = True
        if started:
            pts.append(_xy(wp.transform.location))
        if started and (not wp.is_junction):
            break
    return _dedupe_polyline(pts)


def _tangent(pts: List[np.ndarray], idx: int) -> np.ndarray:
    if len(pts) == 1:
        return np.array([1.0, 0.0], dtype=np.float32)
    if idx == 0:
        v = pts[1] - pts[0]
    elif idx == len(pts) - 1:
        v = pts[-1] - pts[-2]
    else:
        v = pts[idx + 1] - pts[idx - 1]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def _normal_from_tangent(t: np.ndarray) -> np.ndarray:
    return np.array([-float(t[1]), float(t[0])], dtype=np.float32)


@dataclass
class PhaseSplit:
    approach_decide_pts: List[np.ndarray]
    clear_pts: List[np.ndarray]
    touch_idx: Optional[int]


class SimpleWaypoint:
    def __init__(self, x: float, y: float, yaw_rad: float, z: float = 0.0):
        self.transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=float(z)),
            carla.Rotation(pitch=0.0, yaw=math.degrees(yaw_rad), roll=0.0),
        )


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw_rad: float
    speed_kmh: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class CarlaPPOEnvTouchPhase(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario_name: str = "left_turn",
        host: str = "127.0.0.1",
        port: int = 2000,
        tm_port: int = 8000,
        seed: int = 0,
        timeout_s: float = 60.0,
        dest_radius_m: float = 5.0,
        no_rendering: bool = False,
        debug_draw: bool = True,
        debug_draw_interval_steps: int = 10,
        debug_draw_lifetime_s: float = 1.0,
        width_scale: float = 1.0,
        ego_half_width_m: float = 1.0,
        obs_vehicle_count: int = 4,
        obs_radius_m: float = 60.0,
    ):
        super().__init__()
        if scenario_name not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario_name={scenario_name}")

        self.scenario_name = scenario_name
        self.host = host
        self.port = int(port)
        self.tm_port = int(tm_port)
        self.seed_value = int(seed)
        self.timeout_s = float(timeout_s)
        self.dest_radius_m = float(dest_radius_m)
        self.no_rendering = bool(no_rendering)

        self.debug_draw = bool(debug_draw)
        self.debug_draw_interval_steps = max(1, int(debug_draw_interval_steps))
        self.debug_draw_lifetime_s = float(debug_draw_lifetime_s)

        self.width_scale = float(width_scale)
        self.ego_half_width_m = float(ego_half_width_m)
        self.obs_vehicle_count = int(obs_vehicle_count)
        self.obs_radius_m = float(obs_radius_m)

        self.route_sampling_resolution = 2.0
        self.local_anchor_distances_m = [8.0, 18.0, 30.0]
        self.max_lateral_offset_m = 1.0
        self.max_heading_bias_deg = 15.0
        self.min_target_speed_kmh = 0.0
        self.max_target_speed_kmh = 45.0
        self.local_traj_point_spacing_m = 1.5
        self.max_preview_distance_m = 55.0
        self.track_preview_m = 8.0

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(60.0)

        self.handles = None
        self.world = None
        self.dt = 0.05
        self.collision_events: List[Any] = []
        self.episode_start_t: float = 0.0
        self.prev_goal_distance: Optional[float] = None

        self.route_waypoints: List[carla.Waypoint] = []
        self.route_points_xy: List[np.ndarray] = []
        self.route_index = 0

        self.oncoming_route_wps: List[carla.Waypoint] = []
        self.conflict_center_xy: List[np.ndarray] = []
        self.conflict_width_m: float = 4.0

        self.phase_name = "APPROACH_DECIDE"
        self.phase_touch_idx: Optional[int] = None
        self.phase_split = PhaseSplit([], [], None)
        # decision-point timing geometry
        self.decision_point_xy: Optional[np.ndarray] = None
        self.decision_point_route_idx: Optional[int] = None
        self.decision_start_xy: Optional[np.ndarray] = None
        self.decision_start_route_idx: Optional[int] = None
        self.decision_zone_backtrack_m = 10.0
        self.decision_monitor_distance_m = 12.0   # start using timing logic when ego is within this range
        self.decision_reached_tol_m = 2.0
        self.last_decision_active = False
        self.last_dist_to_decision_m = 999.0

        # cache for simple traffic timing features
        self.last_gap_margin_s = 0.0
        self.last_lead_clear_time_s = 999.0
        self.last_follow_arrival_time_s = 999.0
        self.last_ego_arrival_time_s = 999.0
        self.last_follow_closing_speed_mps = 0.0
        self.last_lead_length_m = 0.0

        # keep legacy fields for compatibility, but do not use a hard first-pass gate.
        self.require_first_oncoming_pass = False
        self.first_pass_max_wait_s = 0.0
        self.first_pass_seen_approach = False
        self.first_pass_completed = False
        self.first_pass_wait_elapsed_s = 0.0
        self.first_pass_last_signed_dist_m: Optional[float] = None
        self.first_pass_release_no_vehicle = True

        # learned timing caches: top-3 upcoming conflict-relevant vehicles ordered by ETA to the
        # conflict entry / decision point. These are intended to let the policy infer which gap is
        # usable from continuous traffic, instead of reading a hand-coded gate decision.
        self.last_upcoming_etas_s = [999.0, 999.0, 999.0]
        self.last_upcoming_speeds_mps = [0.0, 0.0, 0.0]
        self.last_upcoming_signed_dist_m = [999.0, 999.0, 999.0]
        self.last_gap12_s = 999.0
        self.last_gap23_s = 999.0

        self.pid_controller: Optional[VehiclePIDController] = None
        self.local_trajectory: List[TrajectoryPoint] = []
        self.last_preview_target: Optional[TrajectoryPoint] = None
        self.last_target_speed_kmh = 20.0
        self.last_lateral_offsets_m = [0.0] * len(self.local_anchor_distances_m)
        self.last_heading_bias_deg = 0.0

        self.step_count = 0

        # 3 lateral offsets + 1 heading bias + 1 target speed
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # 16 ego/route/phase features + 8 per nearby vehicle + 9 conflict-flow timing features
        obs_dim = 16 + self.obs_vehicle_count * 8 + 9
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._build_scene()


    def _active_traffic(self) -> List[carla.Vehicle]:
        if self.handles is None or self.handles.traffic_stream is None:
            return []
        return [v for v in self.handles.traffic_stream.vehicles if v.is_alive]


    def _vehicle_half_extents(self, vehicle: carla.Vehicle) -> Tuple[float, float]:
        bb = vehicle.bounding_box.extent
        return float(bb.x), float(bb.y)

    def _nearest_forward_conflict(self) -> Tuple[Optional[carla.Vehicle], float, float]:
        ego_tf = self.handles.ego.get_transform()
        ego_xy = _xy(ego_tf.location)
        ego_yaw = math.radians(float(ego_tf.rotation.yaw))
        ego_vel = self.handles.ego.get_velocity()
        c = math.cos(-ego_yaw)
        s = math.sin(-ego_yaw)

        best_vehicle: Optional[carla.Vehicle] = None
        best_rel_x = float("inf")
        best_ttc = float("inf")

        for vehicle in self._active_traffic():
            if vehicle.id == self.handles.ego.id:
                continue
            other_xy = _xy(vehicle.get_location())
            rel_world = other_xy - ego_xy
            rel_x = float(c * rel_world[0] - s * rel_world[1])
            rel_y = float(s * rel_world[0] + c * rel_world[1])
            if rel_x <= 0.0 or abs(rel_y) > 5.0:
                continue

            other_vel = vehicle.get_velocity()
            rel_v_world = np.array([other_vel.x - ego_vel.x, other_vel.y - ego_vel.y], dtype=np.float32)
            rel_vx = float(c * rel_v_world[0] - s * rel_v_world[1])
            closing_speed = -rel_vx
            ttc = float("inf")
            if closing_speed > 1e-3:
                ttc = rel_x / closing_speed

            if rel_x < best_rel_x:
                best_vehicle = vehicle
                best_rel_x = rel_x
                best_ttc = ttc

        if best_vehicle is None:
            return None, float("inf"), float("inf")
        return best_vehicle, best_rel_x, best_ttc

    def _neighbor_features(self, vehicle: carla.Vehicle) -> Tuple[float, ...]:
        ego_tf = self.handles.ego.get_transform()
        ego_xy = _xy(ego_tf.location)
        ego_heading = math.radians(float(ego_tf.rotation.yaw))

        veh_xy = _xy(vehicle.get_location())
        rel_world = veh_xy - ego_xy
        c = math.cos(-ego_heading)
        s = math.sin(-ego_heading)
        rel_x = float(c * rel_world[0] - s * rel_world[1])
        rel_y = float(s * rel_world[0] + c * rel_world[1])

        ego_vel = self.handles.ego.get_velocity()
        veh_vel = vehicle.get_velocity()
        rel_v_world = np.array([veh_vel.x - ego_vel.x, veh_vel.y - ego_vel.y], dtype=np.float32)
        rel_vx = float(c * rel_v_world[0] - s * rel_v_world[1])
        rel_vy = float(s * rel_v_world[0] + c * rel_v_world[1])
        dist = float(np.linalg.norm(rel_world))

        half_len, half_wid = self._vehicle_half_extents(vehicle)
        safety_radius = math.sqrt(half_len * half_len + half_wid * half_wid)

        return (
            float(np.clip(rel_x / self.obs_radius_m, -1.0, 1.0)),
            float(np.clip(rel_y / self.obs_radius_m, -1.0, 1.0)),
            float(np.clip(rel_vx / 20.0, -1.0, 1.0)),
            float(np.clip(rel_vy / 20.0, -1.0, 1.0)),
            float(np.clip(dist / self.obs_radius_m, 0.0, 1.0)),
            float(np.clip((2.0 * half_len) / 15.0, 0.0, 1.0)),
            float(np.clip((2.0 * half_wid) / 6.0, 0.0, 1.0)),
            float(np.clip(safety_radius / 8.0, 0.0, 1.0)),
        )

    def _warmup_traffic(self) -> None:
        if self.handles is None or self.handles.traffic_stream is None:
            return
        start_t = self.world.get_snapshot().timestamp.elapsed_seconds
        deadline_t = start_t + max(6.0 + 10.0, 15.0)
        while True:
            sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
            self.handles.traffic_stream.tick(sim_t)
            self.world.tick()
            enough_time = (sim_t - start_t) >= 6.0
            enough_traffic = len(self._active_traffic()) >= 4
            if (enough_time and enough_traffic) or sim_t >= deadline_t:
                return

    def _build_scene(self) -> None:
        preset = SCENARIO_PRESETS[self.scenario_name]
        self.handles = build_scene(
            self.client,
            town=preset["town"],
            seed=self.seed_value,
            ego_spawn_idx=preset["ego_spawn"],
            ego_dest_idx=preset["ego_dest"],
            oncoming_anchor_idx=preset["oncoming_anchor"],
            oncoming_dest_idx=preset["oncoming_dest"],
            n_oncoming=preset["n_oncoming"],
            traffic_profile=preset["traffic_profile"],
            hide_layers=True,
            freeze_lights=True,
            use_stream=True,
            mean_headway_s=preset["mean_headway"],
            burst_prob=preset["burst_prob"],
        )
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        if hasattr(settings, "no_rendering_mode"):
            settings.no_rendering_mode = self.no_rendering
            self.world.apply_settings(settings)
            settings = self.world.get_settings()
        self.dt = settings.fixed_delta_seconds or 0.05

        self.collision_events = []
        self.handles.collision_sensor.listen(lambda event: self.collision_events.append(event))
        self.world.tick()

        self._reset_guidance_stack()
        self._build_fixed_geometry()

    def _reset_guidance_stack(self) -> None:
        self.pid_controller = VehiclePIDController(
            self.handles.ego,
            args_lateral={"K_P": 1.35, "K_I": 0.00, "K_D": 0.25, "dt": self.dt},
            args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": self.dt},
            max_throttle=0.75,
            max_brake=0.5,
            max_steering=0.8,
        )

    def _build_fixed_geometry(self) -> None:
        carla_map = self.world.get_map()
        grp = GlobalRoutePlanner(carla_map, self.route_sampling_resolution)

        ego_start = self.handles.ego_start.location
        ego_dest_wp = carla_map.get_waypoint(
            self.handles.ego_destination,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        ego_plan = grp.trace_route(ego_start, ego_dest_wp.transform.location)
        self.route_waypoints = [wp for wp, _ in ego_plan]
        self.route_points_xy = [_xy(wp.transform.location) for wp in self.route_waypoints]

        # oncoming route: take points the same way as ego
        preset = SCENARIO_PRESETS[self.scenario_name]
        spawn_points = carla_map.get_spawn_points()
        oncoming_start = self.handles.oncoming_anchor.location
        oncoming_dest = spawn_points[preset["oncoming_dest"]].location
        oncoming_plan = grp.trace_route(oncoming_start, oncoming_dest)
        self.oncoming_route_wps = [wp for wp, _ in oncoming_plan]
        self.conflict_center_xy = _extract_junction_route_xy(self.oncoming_route_wps)
        self.conflict_width_m = (
            float(self.oncoming_route_wps[0].lane_width) * self.width_scale
            if self.oncoming_route_wps else 4.0
        )

        ego_pts = list(self.route_points_xy)
        phase_split = self._split_route_by_touch(ego_pts, self.conflict_center_xy, self.conflict_width_m)
        self.phase_split = phase_split
        self.phase_touch_idx = phase_split.touch_idx
        self.decision_point_xy, self.decision_point_route_idx = self._compute_decision_point(
            ego_pts, self.conflict_center_xy, self.conflict_width_m
        )
        if self.decision_point_route_idx is not None:
            self.decision_start_route_idx = self._route_index_backtrack_distance(
                self.decision_point_route_idx, self.decision_zone_backtrack_m
            )
            self.decision_start_xy = self.route_points_xy[self.decision_start_route_idx].copy()
        else:
            self.decision_start_route_idx = None
            self.decision_start_xy = None



    def _route_index_backtrack_distance(self, end_idx: Optional[int], backtrack_m: float) -> int:
        if not self.route_points_xy:
            return 0
        if end_idx is None:
            return 0
        idx = max(0, min(end_idx, len(self.route_points_xy) - 1))
        acc = 0.0
        while idx > 0 and acc < backtrack_m:
            acc += float(np.linalg.norm(self.route_points_xy[idx] - self.route_points_xy[idx - 1]))
            idx -= 1
        return idx

    def _is_decision_active(self, ego_xy: np.ndarray, in_conflict: bool) -> bool:
        if in_conflict or self.decision_point_xy is None or self.decision_start_route_idx is None:
            self.last_dist_to_decision_m = 999.0 if self.decision_point_xy is None else float(np.linalg.norm(self.decision_point_xy - ego_xy))
            self.last_decision_active = False
            return False

        self._advance_route_index()
        self.last_dist_to_decision_m = float(np.linalg.norm(self.decision_point_xy - ego_xy))
        in_decision_segment = self.route_index >= self.decision_start_route_idx
        near_decision = self.last_dist_to_decision_m <= self.decision_monitor_distance_m
        active = (in_decision_segment or near_decision) and (self.phase_name != "CLEAR")
        self.last_decision_active = bool(active)
        return bool(active)

    def _polyline_arc_length_between(self, pts: List[np.ndarray], i0: int, i1: int) -> float:
        if not pts or i0 is None or i1 is None:
            return 0.0
        i0 = max(0, min(i0, len(pts) - 1))
        i1 = max(0, min(i1, len(pts) - 1))
        if i1 <= i0:
            return 0.0

        s = 0.0
        for i in range(i0, i1):
            s += float(np.linalg.norm(pts[i + 1] - pts[i]))
        return s

    def _nearest_polyline_index(self, pt: np.ndarray, poly: List[np.ndarray]) -> int:
        if not poly:
            return 0
        best_i = 0
        best_d = float("inf")
        for i, p in enumerate(poly):
            d = float(np.linalg.norm(pt - p))
            if d < best_d:
                best_d = d
                best_i = i
        return best_i


    def _nearest_oncoming_signed_distance_to_decision(self) -> Tuple[Optional[float], Optional[float]]:
        if self.decision_point_xy is None or not self.conflict_center_xy:
            return None, None

        dp_idx = self._nearest_polyline_index(self.decision_point_xy, self.conflict_center_xy)
        best_signed_dist: Optional[float] = None
        best_speed_mps: Optional[float] = None

        for vehicle in self._active_traffic():
            veh_xy = _xy(vehicle.get_location())
            veh_idx = self._nearest_polyline_index(veh_xy, self.conflict_center_xy)

            if veh_idx <= dp_idx:
                dist_to_dp = self._polyline_arc_length_between(self.conflict_center_xy, veh_idx, dp_idx)
                signed_dist = +dist_to_dp
            else:
                dist_past_dp = self._polyline_arc_length_between(self.conflict_center_xy, dp_idx, veh_idx)
                signed_dist = -dist_past_dp

            if best_signed_dist is None or abs(signed_dist) < abs(best_signed_dist):
                best_signed_dist = float(signed_dist)
                best_speed_mps = float(_speed_kmh(vehicle) / 3.6)

        return best_signed_dist, best_speed_mps

    def _update_first_pass_gate(self, decision_active: bool) -> None:
        if not self.require_first_oncoming_pass:
            self.first_pass_completed = True
            self.first_pass_release_no_vehicle = True
            return

        if not decision_active:
            self.first_pass_wait_elapsed_s = 0.0
            self.first_pass_last_signed_dist_m = None
            return

        self.first_pass_wait_elapsed_s += float(self.dt)
        signed_dist, _ = self._nearest_oncoming_signed_distance_to_decision()

        if signed_dist is not None:
            if signed_dist > 0.0:
                self.first_pass_seen_approach = True

            prev = self.first_pass_last_signed_dist_m
            if self.first_pass_seen_approach and signed_dist <= 0.0:
                self.first_pass_completed = True

            if prev is not None and prev > 0.0 and signed_dist <= 0.0:
                self.first_pass_completed = True

            self.first_pass_last_signed_dist_m = signed_dist
        else:
            self.first_pass_last_signed_dist_m = None
            if self.first_pass_wait_elapsed_s >= self.first_pass_max_wait_s:
                self.first_pass_release_no_vehicle = True
                self.first_pass_completed = True

    def _decision_gate_open(self) -> bool:
        if not self.require_first_oncoming_pass:
            return True
        return bool(self.first_pass_completed or self.first_pass_release_no_vehicle)

    def _decision_timing_features(self) -> Dict[str, float]:
        if self.decision_point_xy is None or not self.conflict_center_xy:
            return {
                "lead_clear_time_s": 999.0,
                "follow_arrival_time_s": 999.0,
                "ego_arrival_time_s": 999.0,
                "gap_margin_s": -999.0,
                "follow_closing_speed_mps": 0.0,
                "lead_length_m": 0.0,
            }

        dp = self.decision_point_xy
        ego = self.handles.ego

        # ego time to decision point
        ego_xy = _xy(ego.get_location())
        ego_speed_mps = _speed_kmh(ego) / 3.6
        ego_dist = float(np.linalg.norm(dp - ego_xy))
        ego_arrival_time_s = ego_dist / max(ego_speed_mps, 1.0)

        # approximate decision point index on oncoming conflict centerline
        dp_idx = self._nearest_polyline_index(dp, self.conflict_center_xy)

        lead_clear_time_s = 999.0
        follow_arrival_time_s = 999.0
        follow_closing_speed_mps = 0.0
        lead_length_m = 0.0

        best_lead_dist = -float("inf")   # already passed dp, choose nearest passed
        best_follow_dist = float("inf")  # not yet reached dp, choose nearest approaching

        for vehicle in self._active_traffic():
            veh_xy = _xy(vehicle.get_location())
            veh_idx = self._nearest_polyline_index(veh_xy, self.conflict_center_xy)

            # signed arc-distance relative to decision point on conflict centerline
            if veh_idx <= dp_idx:
                dist_to_dp = self._polyline_arc_length_between(self.conflict_center_xy, veh_idx, dp_idx)
                signed_dist = +dist_to_dp   # approaching
            else:
                dist_past_dp = self._polyline_arc_length_between(self.conflict_center_xy, dp_idx, veh_idx)
                signed_dist = -dist_past_dp  # already passed

            veh_speed = _speed_kmh(vehicle) / 3.6
            half_len, half_wid = self._vehicle_half_extents(vehicle)
            veh_length = 2.0 * half_len

            # lead vehicle: just passed decision point
            if signed_dist < 0.0 and signed_dist > best_lead_dist:
                best_lead_dist = signed_dist
                # vehicle tail fully clears after additional half length
                lead_clear_time_s = (abs(signed_dist) + veh_length) / max(veh_speed, 1.0)
                lead_length_m = veh_length

            # follower: nearest one still approaching
            if signed_dist >= 0.0 and signed_dist < best_follow_dist:
                best_follow_dist = signed_dist
                follow_arrival_time_s = signed_dist / max(veh_speed, 1.0)

                # approximate closing speed toward decision point
                vel = vehicle.get_velocity()
                tangent = np.array([1.0, 0.0], dtype=np.float32)
                if len(self.conflict_center_xy) >= 2:
                    idx0 = max(0, min(veh_idx, len(self.conflict_center_xy) - 2))
                    seg = self.conflict_center_xy[idx0 + 1] - self.conflict_center_xy[idx0]
                    n = float(np.linalg.norm(seg))
                    if n > 1e-6:
                        tangent = (seg / n).astype(np.float32)
                vxy = np.array([vel.x, vel.y], dtype=np.float32)
                follow_closing_speed_mps = float(np.dot(vxy, tangent))

        gap_margin_s = follow_arrival_time_s - max(lead_clear_time_s, ego_arrival_time_s)

        return {
            "lead_clear_time_s": float(lead_clear_time_s),
            "follow_arrival_time_s": float(follow_arrival_time_s),
            "ego_arrival_time_s": float(ego_arrival_time_s),
            "gap_margin_s": float(gap_margin_s),
            "follow_closing_speed_mps": float(follow_closing_speed_mps),
            "lead_length_m": float(lead_length_m),
        }

    def _rank_upcoming_conflict_vehicles(self) -> List[Dict[str, float]]:
        if self.decision_point_xy is None or not self.conflict_center_xy:
            return []

        dp_idx = self._nearest_polyline_index(self.decision_point_xy, self.conflict_center_xy)
        ranked: List[Dict[str, float]] = []
        for vehicle in self._active_traffic():
            veh_xy = _xy(vehicle.get_location())
            veh_idx = self._nearest_polyline_index(veh_xy, self.conflict_center_xy)
            if veh_idx <= dp_idx:
                signed_dist = self._polyline_arc_length_between(self.conflict_center_xy, veh_idx, dp_idx)
            else:
                signed_dist = -self._polyline_arc_length_between(self.conflict_center_xy, dp_idx, veh_idx)

            speed_mps = max(_speed_kmh(vehicle) / 3.6, 0.1)
            if signed_dist < -5.0:
                continue
            eta_s = signed_dist / speed_mps if signed_dist >= 0.0 else -abs(signed_dist) / speed_mps
            ranked.append({
                "vehicle_id": float(vehicle.id),
                "signed_dist_m": float(signed_dist),
                "eta_s": float(eta_s),
                "speed_mps": float(speed_mps),
            })

        ranked.sort(key=lambda item: item["eta_s"])
        return ranked

    def _update_conflict_flow_cache(self, decision_active: bool) -> None:
        ranked = self._rank_upcoming_conflict_vehicles() if decision_active else []
        etas = [999.0, 999.0, 999.0]
        speeds = [0.0, 0.0, 0.0]
        dists = [999.0, 999.0, 999.0]
        upcoming = [r for r in ranked if r["eta_s"] >= 0.0][:3]
        for i, item in enumerate(upcoming):
            etas[i] = float(item["eta_s"])
            speeds[i] = float(item["speed_mps"])
            dists[i] = float(item["signed_dist_m"])
        self.last_upcoming_etas_s = etas
        self.last_upcoming_speeds_mps = speeds
        self.last_upcoming_signed_dist_m = dists
        self.last_gap12_s = etas[1] - etas[0] if etas[0] < 900.0 and etas[1] < 900.0 else 999.0
        self.last_gap23_s = etas[2] - etas[1] if etas[1] < 900.0 and etas[2] < 900.0 else 999.0

    def _find_first_conflict_touch_idx(
        self,
        ego_pts: List[np.ndarray],
        conflict_center_xy: List[np.ndarray],
        conflict_width: float,
    ) -> Optional[int]:
        if not ego_pts or not conflict_center_xy:
            return None
        touch_dist_th = 0.5 * conflict_width + self.ego_half_width_m
        for i, p in enumerate(ego_pts):
            d = _distance_point_to_polyline(p, conflict_center_xy)
            if d <= touch_dist_th:
                return i
        return None

    def _split_route_by_touch(
        self,
        ego_pts: List[np.ndarray],
        conflict_center_xy: List[np.ndarray],
        conflict_width: float,
    ) -> PhaseSplit:
        touch_idx = self._find_first_conflict_touch_idx(ego_pts, conflict_center_xy, conflict_width)
        if touch_idx is None:
            return PhaseSplit(approach_decide_pts=ego_pts, clear_pts=[], touch_idx=None)
        return PhaseSplit(
            approach_decide_pts=ego_pts[:touch_idx],
            clear_pts=ego_pts[touch_idx:],
            touch_idx=touch_idx,
        )

    def _compute_decision_point(
        self,
        ego_pts: List[np.ndarray],
        conflict_center_xy: List[np.ndarray],
        conflict_width: float,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        if not ego_pts or not conflict_center_xy:
            return None, None

        touch_dist_th = 0.5 * conflict_width + self.ego_half_width_m

        for i in range(len(ego_pts)):
            p = ego_pts[i]
            d = _distance_point_to_polyline(p, conflict_center_xy)
            if d <= touch_dist_th:
                return p.copy(), i

        return None, None

    def _update_phase(self) -> None:
        ego_xy = _xy(self.handles.ego.get_location())
        self._advance_route_index()
        if self.conflict_center_xy:
            d = _distance_point_to_polyline(ego_xy, self.conflict_center_xy)
            touch_dist_th = 0.5 * self.conflict_width_m + self.ego_half_width_m
            in_conflict = d <= touch_dist_th
            if in_conflict:
                self.phase_name = "CLEAR"
            elif self.decision_start_route_idx is not None and self.route_index >= self.decision_start_route_idx:
                self.phase_name = "DECISION"
            else:
                self.phase_name = "APPROACH_DECIDE"
        else:
            self.phase_name = "APPROACH_DECIDE"

    def _decode_action(self, action: np.ndarray) -> Tuple[List[float], float, float]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        lateral_offsets = [
            float(np.clip(a, -1.0, 1.0) * self.max_lateral_offset_m)
            for a in action[: len(self.local_anchor_distances_m)]
        ]
        heading_bias_deg = float(np.clip(action[len(self.local_anchor_distances_m)], -1.0, 1.0) * self.max_heading_bias_deg)
        speed_action = float(np.clip(action[len(self.local_anchor_distances_m) + 1], -1.0, 1.0))
        speed = self.min_target_speed_kmh + ((speed_action + 1.0) * 0.5) * (
            self.max_target_speed_kmh - self.min_target_speed_kmh
        )
        speed = float(np.clip(speed, self.min_target_speed_kmh, self.max_target_speed_kmh))
        self.last_lateral_offsets_m = lateral_offsets
        self.last_heading_bias_deg = heading_bias_deg
        self.last_target_speed_kmh = speed
        return lateral_offsets, heading_bias_deg, speed

    def _route_point_at_distance(self, start_idx: int, target_dist_m: float) -> Tuple[np.ndarray, float, int]:
        if not self.route_points_xy:
            ego_xy = _xy(self.handles.ego.get_location())
            heading = math.radians(self.handles.ego.get_transform().rotation.yaw)
            return ego_xy, heading, 0

        acc = 0.0
        idx = max(0, min(start_idx, len(self.route_points_xy) - 1))
        prev = self.route_points_xy[idx]
        while idx + 1 < len(self.route_points_xy) and acc < target_dist_m:
            nxt = self.route_points_xy[idx + 1]
            seg = float(np.linalg.norm(nxt - prev))
            if acc + seg >= target_dist_m and seg > 1e-6:
                ratio = (target_dist_m - acc) / seg
                pt = prev + ratio * (nxt - prev)
                heading = math.atan2(float((nxt - prev)[1]), float((nxt - prev)[0]))
                return pt.astype(np.float32), heading, idx + 1
            acc += seg
            idx += 1
            prev = nxt

        if len(self.route_points_xy) >= 2:
            p0 = self.route_points_xy[max(0, idx - 1)]
            p1 = self.route_points_xy[idx]
            heading = math.atan2(float((p1 - p0)[1]), float((p1 - p0)[0])) if float(np.linalg.norm(p1 - p0)) > 1e-6 else 0.0
        else:
            heading = 0.0
        return self.route_points_xy[idx].astype(np.float32), heading, idx

    def _advance_route_index(self) -> None:
        if not self.route_points_xy:
            self.route_index = 0
            return
        ego_xy = _xy(self.handles.ego.get_location())
        upper = min(len(self.route_points_xy), self.route_index + 30)
        best_idx = self.route_index
        best_dist = float("inf")
        for idx in range(self.route_index, upper):
            dist = float(np.linalg.norm(ego_xy - self.route_points_xy[idx]))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        self.route_index = best_idx

    def _catmull_rom_chain(self, pts: List[np.ndarray], samples_per_seg: int = 8) -> List[np.ndarray]:
        if len(pts) < 2:
            return list(pts)
        extended = [pts[0]] + list(pts) + [pts[-1]]
        result: List[np.ndarray] = []
        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
            for j in range(samples_per_seg):
                t = j / float(samples_per_seg)
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                result.append(point.astype(np.float32))
        result.append(np.asarray(pts[-1], dtype=np.float32))
        return result

    def _trajectory_points_from_action(self, action: np.ndarray) -> List[TrajectoryPoint]:
        lateral_offsets_m, heading_bias_deg, target_speed_kmh = self._decode_action(action)
        self._advance_route_index()

        ego_tf = self.handles.ego.get_transform()
        ego_xy = _xy(ego_tf.location)

        control_pts: List[np.ndarray] = [ego_xy]
        reconnect_idx = self.route_index
        for d_m, lat_m in zip(self.local_anchor_distances_m, lateral_offsets_m):
            center_pt, tangent_heading, reconnect_idx = self._route_point_at_distance(self.route_index, d_m)
            normal = _normal_from_tangent(np.array([math.cos(tangent_heading), math.sin(tangent_heading)], dtype=np.float32))
            offset_pt = center_pt + normal * lat_m
            control_pts.append(offset_pt.astype(np.float32))

        reconnect_pt, reconnect_heading, reconnect_idx = self._route_point_at_distance(
            self.route_index, min(self.max_preview_distance_m, self.local_anchor_distances_m[-1] + 10.0)
        )
        reconnect_heading += math.radians(heading_bias_deg)
        reconnect_pt = reconnect_pt + _normal_from_tangent(
            np.array([math.cos(reconnect_heading), math.sin(reconnect_heading)], dtype=np.float32)
        ) * lateral_offsets_m[-1]
        control_pts.append(reconnect_pt.astype(np.float32))

        spline_pts = self._catmull_rom_chain(control_pts, samples_per_seg=8)
        if len(spline_pts) < 2:
            spline_pts = control_pts

        dense_pts: List[np.ndarray] = [spline_pts[0]]
        for p in spline_pts[1:]:
            seg_len = float(np.linalg.norm(dense_pts[-1] - p))
            if seg_len < 0.5:
                continue
            n_insert = max(1, int(seg_len / self.local_traj_point_spacing_m))
            start = dense_pts[-1]
            for k in range(1, n_insert + 1):
                alpha = k / float(n_insert)
                dense_pts.append((start + alpha * (p - start)).astype(np.float32))

        trajectory: List[TrajectoryPoint] = []
        if len(dense_pts) == 1:
            dense_pts.append(dense_pts[0] + np.array([0.5, 0.0], dtype=np.float32))

        for i, pt in enumerate(dense_pts):
            if i == 0:
                nxt = dense_pts[1]
                yaw = math.atan2(float((nxt - pt)[1]), float((nxt - pt)[0]))
            elif i == len(dense_pts) - 1:
                prv = dense_pts[i - 1]
                yaw = math.atan2(float((pt - prv)[1]), float((pt - prv)[0]))
            else:
                prv = dense_pts[i - 1]
                nxt = dense_pts[i + 1]
                yaw = math.atan2(float((nxt - prv)[1]), float((nxt - prv)[0]))
            trajectory.append(TrajectoryPoint(float(pt[0]), float(pt[1]), yaw, target_speed_kmh))

        self.local_trajectory = trajectory
        return trajectory

    def _nearest_trajectory_index(self, ego_xy: np.ndarray, search_limit: int = 60) -> int:
        if not self.local_trajectory:
            return 0
        upper = min(len(self.local_trajectory), search_limit)
        best_idx = 0
        best_dist = float("inf")
        for idx in range(upper):
            pt = self.local_trajectory[idx].as_np()
            d = float(np.linalg.norm(ego_xy - pt))
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _tracking_target(self) -> TrajectoryPoint:
        ego_xy = _xy(self.handles.ego.get_location())
        if not self.local_trajectory:
            ego_heading = math.radians(self.handles.ego.get_transform().rotation.yaw)
            return TrajectoryPoint(float(ego_xy[0]), float(ego_xy[1]), ego_heading, self.last_target_speed_kmh)

        nearest_idx = self._nearest_trajectory_index(ego_xy)
        accum = 0.0
        for idx in range(nearest_idx, len(self.local_trajectory) - 1):
            p0 = self.local_trajectory[idx].as_np()
            p1 = self.local_trajectory[idx + 1].as_np()
            accum += float(np.linalg.norm(p1 - p0))
            if accum >= self.track_preview_m:
                return self.local_trajectory[idx + 1]
        return self.local_trajectory[-1]

    def _trajectory_waypoint_proxy(self, target: TrajectoryPoint) -> SimpleWaypoint:
        ego_z = self.handles.ego.get_location().z
        return SimpleWaypoint(target.x, target.y, target.yaw_rad, z=ego_z)

    def _goal_distance(self) -> float:
        return self.handles.ego.get_location().distance(self.handles.ego_destination)


    def _get_obs(self) -> np.ndarray:
        ego = self.handles.ego
        speed = _speed_kmh(ego)
        goal_distance = self._goal_distance()
        ego_xy = _xy(ego.get_location())
        self._update_phase()

        track_target = self._tracking_target()
        track_delta = track_target.as_np() - ego_xy
        track_dist = float(np.linalg.norm(track_delta))
        track_heading_error = track_target.yaw_rad - math.radians(ego.get_transform().rotation.yaw)

        dist_to_conflict = _distance_point_to_polyline(ego_xy, self.conflict_center_xy) if self.conflict_center_xy else 999.0
        touch_dist_th = 0.5 * self.conflict_width_m + self.ego_half_width_m
        in_conflict = 1.0 if dist_to_conflict <= touch_dist_th else 0.0

        decision_active = self._is_decision_active(ego_xy, bool(in_conflict))
        if decision_active:
            timing = self._decision_timing_features()
            self.last_lead_clear_time_s = timing["lead_clear_time_s"]
            self.last_follow_arrival_time_s = timing["follow_arrival_time_s"]
            self.last_ego_arrival_time_s = timing["ego_arrival_time_s"]
            self.last_gap_margin_s = timing["gap_margin_s"]
            self.last_follow_closing_speed_mps = timing["follow_closing_speed_mps"]
            self.last_lead_length_m = timing["lead_length_m"]
        else:
            self.last_lead_clear_time_s = 999.0
            self.last_follow_arrival_time_s = 999.0
            self.last_ego_arrival_time_s = 999.0
            self.last_gap_margin_s = 0.0
            self.last_follow_closing_speed_mps = 0.0
            self.last_lead_length_m = 0.0
        self._update_conflict_flow_cache(decision_active)

        phase_map = {"APPROACH_DECIDE": 0.0, "DECISION": 0.5, "CLEAR": 1.0}
        phase_code = phase_map.get(self.phase_name, 0.0)

        _, forward_headway, forward_ttc = self._nearest_forward_conflict()
        nearby_count = 0
        others: List[Tuple[float, carla.Vehicle]] = []
        for vehicle in self._active_traffic():
            if vehicle.id == ego.id:
                continue
            dist = ego.get_location().distance(vehicle.get_location())
            if dist <= self.obs_radius_m:
                others.append((dist, vehicle))
        others.sort(key=lambda item: item[0])
        nearby_count = len(others)

        obs: List[float] = [
            float(np.clip(speed / max(self.max_target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(goal_distance / 120.0, 0.0, 1.0)),
            float(np.sin(track_heading_error)),
            float(np.cos(track_heading_error)),
            float(np.clip(track_dist / self.max_preview_distance_m, 0.0, 1.0)),
            float(np.clip(self.last_target_speed_kmh / max(self.max_target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(np.mean(self.last_lateral_offsets_m) / max(self.max_lateral_offset_m, 1e-3), -1.0, 1.0)),
            float(np.clip(self.last_heading_bias_deg / max(self.max_heading_bias_deg, 1e-3), -1.0, 1.0)),
            float(np.clip(dist_to_conflict / 25.0, 0.0, 1.0)),
            float(in_conflict),
            float(phase_code),
            float(np.clip(self.conflict_width_m / 8.0, 0.0, 1.0)),
            float(np.clip(float(self.route_index) / max(len(self.route_points_xy) - 1, 1), 0.0, 1.0)),
            float(np.clip(forward_headway / self.obs_radius_m, 0.0, 1.0)) if math.isfinite(forward_headway) else 1.0,
            float(np.clip(forward_ttc / 10.0, 0.0, 1.0)) if math.isfinite(forward_ttc) else 1.0,
            float(np.clip(nearby_count / max(self.obs_vehicle_count, 1), 0.0, 1.0)),
        ]

        for _, vehicle in others[: self.obs_vehicle_count]:
            obs.extend(self._neighbor_features(vehicle))
        while len(obs) < 16 + self.obs_vehicle_count * 8:
            obs.extend([0.0] * 8)

        obs.extend([
            float(np.clip(self.last_ego_arrival_time_s / 10.0, 0.0, 1.0)) if decision_active else 1.0,
            float(np.clip(self.last_upcoming_etas_s[0] / 10.0, 0.0, 1.0)),
            float(np.clip(self.last_upcoming_etas_s[1] / 10.0, 0.0, 1.0)),
            float(np.clip(self.last_upcoming_etas_s[2] / 10.0, 0.0, 1.0)),
            float(np.clip(self.last_upcoming_speeds_mps[0] / 20.0, 0.0, 1.0)),
            float(np.clip(self.last_upcoming_speeds_mps[1] / 20.0, 0.0, 1.0)),
            float(np.clip(self.last_upcoming_speeds_mps[2] / 20.0, 0.0, 1.0)),
            float(np.clip(self.last_gap12_s / 5.0, 0.0, 1.0)) if self.last_gap12_s < 900.0 else 1.0,
            float(np.clip(self.last_gap23_s / 5.0, 0.0, 1.0)) if self.last_gap23_s < 900.0 else 1.0,
        ])
        assert len(obs) == self.observation_space.shape[0], (len(obs), self.observation_space.shape[0])
        return np.asarray(obs, dtype=np.float32)

    def _draw_polyline(self, pts: List[np.ndarray], z: float, color: carla.Color, thickness: float) -> None:
        if len(pts) < 2:
            return
        dbg = self.world.debug
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            dbg.draw_line(
                carla.Location(x=float(p0[0]), y=float(p0[1]), z=z),
                carla.Location(x=float(p1[0]), y=float(p1[1]), z=z),
                thickness=thickness,
                color=color,
                life_time=self.debug_draw_lifetime_s,
                persistent_lines=False,
            )

    def _draw_strip(self, center_pts: List[np.ndarray], width_m: float, z: float) -> None:
        if not center_pts:
            return
        half_w = 0.5 * width_m
        left_pts: List[np.ndarray] = []
        right_pts: List[np.ndarray] = []
        dbg = self.world.debug
        for i, p in enumerate(center_pts):
            t = _tangent(center_pts, i)
            n = _normal_from_tangent(t)
            left_pts.append((p + n * half_w).astype(np.float32))
            right_pts.append((p - n * half_w).astype(np.float32))

        self._draw_polyline(center_pts, z, carla.Color(255, 0, 0), 0.12)
        self._draw_polyline(left_pts, z, carla.Color(255, 128, 128), 0.08)
        self._draw_polyline(right_pts, z, carla.Color(255, 128, 128), 0.08)

        if len(center_pts) >= 2:
            step = max(1, len(center_pts) // 10)
            for i in range(0, len(center_pts), step):
                lp, rp = left_pts[i], right_pts[i]
                dbg.draw_line(
                    carla.Location(x=float(lp[0]), y=float(lp[1]), z=z),
                    carla.Location(x=float(rp[0]), y=float(rp[1]), z=z),
                    thickness=0.05,
                    color=carla.Color(255, 100, 100),
                    life_time=self.debug_draw_lifetime_s,
                    persistent_lines=False,
                )


    def _draw_debug(self) -> None:
        if not self.debug_draw or self.world is None:
            return
        if (self.step_count % self.debug_draw_interval_steps) != 0:
            return

        z = self.handles.ego.get_location().z + 0.3
        self._draw_strip(self.conflict_center_xy, self.conflict_width_m, z + 0.02)

        self._draw_polyline(self.phase_split.approach_decide_pts, z, carla.Color(0, 255, 0), 0.12)
        self._draw_polyline(self.phase_split.clear_pts, z + 0.01, carla.Color(0, 255, 255), 0.14)

        if self.phase_touch_idx is not None and self.phase_touch_idx < len(self.route_points_xy):
            p = self.route_points_xy[self.phase_touch_idx]
            self.world.debug.draw_point(
                carla.Location(x=float(p[0]), y=float(p[1]), z=z + 0.03),
                size=0.18,
                color=carla.Color(255, 0, 0),
                life_time=self.debug_draw_lifetime_s,
                persistent_lines=False,
            )
            self.world.debug.draw_string(
                carla.Location(x=float(p[0]), y=float(p[1]), z=z + 0.16),
                "TOUCH",
                color=carla.Color(255, 0, 0),
                life_time=self.debug_draw_lifetime_s,
            )

        if self.decision_start_xy is not None and self.decision_point_xy is not None:
            ds = self.decision_start_xy
            dp = self.decision_point_xy
            self.world.debug.draw_point(
                carla.Location(x=float(ds[0]), y=float(ds[1]), z=z + 0.04),
                size=0.16,
                color=carla.Color(255, 255, 0),
                life_time=self.debug_draw_lifetime_s,
                persistent_lines=False,
            )
            self.world.debug.draw_string(
                carla.Location(x=float(ds[0]), y=float(ds[1]), z=z + 0.18),
                "DECISION_START",
                color=carla.Color(255, 255, 0),
                life_time=self.debug_draw_lifetime_s,
            )
            self.world.debug.draw_point(
                carla.Location(x=float(dp[0]), y=float(dp[1]), z=z + 0.04),
                size=0.20,
                color=carla.Color(255, 255, 255),
                life_time=self.debug_draw_lifetime_s,
                persistent_lines=False,
            )
            self.world.debug.draw_string(
                carla.Location(x=float(dp[0]), y=float(dp[1]), z=z + 0.24),
                f"DP phase={self.phase_name}",
                color=carla.Color(255, 255, 255),
                life_time=self.debug_draw_lifetime_s,
            )
            self.world.debug.draw_line(
                carla.Location(x=float(ds[0]), y=float(ds[1]), z=z + 0.02),
                carla.Location(x=float(dp[0]), y=float(dp[1]), z=z + 0.02),
                thickness=0.08,
                color=carla.Color(255, 255, 0) if self.last_decision_active else carla.Color(160, 160, 0),
                life_time=self.debug_draw_lifetime_s,
                persistent_lines=False,
            )
            self.world.debug.draw_string(
                carla.Location(x=float(dp[0]), y=float(dp[1]), z=z + 0.40),
                (
                    f"active={int(self.last_decision_active)} "
                    f"gate=1 "
                    f"seen={int(self.first_pass_seen_approach)} "
                    f"seq={self.last_upcoming_etas_s[0]:.1f}/{self.last_upcoming_etas_s[1]:.1f}/{self.last_upcoming_etas_s[2]:.1f} "
                    f"gap={self.last_gap_margin_s:.2f} "
                    f"lead={self.last_lead_clear_time_s:.2f} "
                    f"follow={self.last_follow_arrival_time_s:.2f}"
                ),
                color=carla.Color(255, 255, 0) if self.last_decision_active else carla.Color(180, 180, 180),
                life_time=self.debug_draw_lifetime_s,
            )

        ego_loc = self.handles.ego.get_location()
        self.world.debug.draw_string(
            carla.Location(x=ego_loc.x, y=ego_loc.y, z=ego_loc.z + 1.0),
            f"{self.phase_name} active={int(self.last_decision_active)}",
            color=carla.Color(0, 255, 255) if self.phase_name == "CLEAR" else carla.Color(255, 255, 0) if self.last_decision_active else carla.Color(0, 255, 0),
            life_time=self.debug_draw_lifetime_s,
        )

        if self.local_trajectory:
            self._draw_polyline([p.as_np() for p in self.local_trajectory], z + 0.04, carla.Color(0, 0, 255), 0.10)


    def _compute_reward(self, success: bool, crashed: bool, timed_out: bool) -> Tuple[float, Dict[str, float]]:
        goal_distance = self._goal_distance()
        progress = 0.0
        if self.prev_goal_distance is not None:
            progress = self.prev_goal_distance - goal_distance
        self.prev_goal_distance = goal_distance

        ego_xy = _xy(self.handles.ego.get_location())
        dist_to_conflict = _distance_point_to_polyline(ego_xy, self.conflict_center_xy) if self.conflict_center_xy else 999.0
        touch_dist_th = 0.5 * self.conflict_width_m + self.ego_half_width_m
        in_conflict = dist_to_conflict <= touch_dist_th
        decision_active = self._is_decision_active(ego_xy, in_conflict)
        self._update_conflict_flow_cache(decision_active)

        speed_kmh = _speed_kmh(self.handles.ego)
        control = self.handles.ego.get_control()

        progress_reward = 1.0 * progress
        comfort_penalty = 0.02 * abs(control.steer) + 0.015 * control.brake

        traffic_penalty = 0.0
        ego_loc = self.handles.ego.get_location()
        ego_tf = self.handles.ego.get_transform()
        ego_yaw = math.radians(float(ego_tf.rotation.yaw))
        ego_vel = self.handles.ego.get_velocity()
        ego_vel_xy = np.array([ego_vel.x, ego_vel.y], dtype=np.float32)
        c = math.cos(-ego_yaw)
        s = math.sin(-ego_yaw)
        ego_half_len = self.ego_half_width_m
        ego_half_wid = self.ego_half_width_m
        ego_radius = math.sqrt(ego_half_len * ego_half_len + ego_half_wid * ego_half_wid)

        for vehicle in self._active_traffic():
            if vehicle.id == self.handles.ego.id:
                continue
            other_half_len, other_half_wid = self._vehicle_half_extents(vehicle)
            safe_gap = 1.5 + ego_radius + math.sqrt(other_half_len * other_half_len + other_half_wid * other_half_wid)
            dist = ego_loc.distance(vehicle.get_location())
            if dist < safe_gap:
                traffic_penalty += 0.10 * (safe_gap - dist)

            other_xy = _xy(vehicle.get_location())
            rel_world = other_xy - ego_xy
            rel_x = float(c * rel_world[0] - s * rel_world[1])
            rel_y = float(s * rel_world[0] + c * rel_world[1])
            if rel_x <= 0.0 or abs(rel_y) > 5.0:
                continue

            other_vel = vehicle.get_velocity()
            rel_v_world = np.array([other_vel.x - ego_vel_xy[0], other_vel.y - ego_vel_xy[1]], dtype=np.float32)
            rel_vx = float(c * rel_v_world[0] - s * rel_v_world[1])
            closing_speed = -rel_vx
            if closing_speed <= 1e-3:
                continue
            ttc = rel_x / closing_speed
            if ttc < 3.5:
                traffic_penalty += 0.25 * (3.5 - ttc)

        _, forward_headway, forward_ttc = self._nearest_forward_conflict()
        yield_bonus = 0.0
        timing_bonus = 0.0
        timing_penalty = 0.0

        eta1, eta2, eta3 = self.last_upcoming_etas_s
        ego_eta = self.last_ego_arrival_time_s
        valid1 = eta1 < 900.0
        valid2 = eta2 < 900.0
        valid3 = eta3 < 900.0

        hesitation_penalty = 0.0
        commit_bonus = 0.0
        entry_commit_bonus = 0.0

        if self.phase_name == "APPROACH_DECIDE":
            phase_bonus = 0.0
            phase_penalty = 0.0
            if math.isfinite(forward_ttc) and forward_ttc < 3.8 and speed_kmh < 6.0:
                yield_bonus = 0.015

        elif self.phase_name == "DECISION":
            phase_bonus = 0.0
            phase_penalty = 0.0
            if decision_active:
                unsafe_now = valid1 and ego_eta + 0.6 >= eta1
                unsafe_second = valid2 and ego_eta + 0.3 >= eta2
                good_late_gap = valid2 and valid3 and eta2 < ego_eta - 0.15 and eta3 > ego_eta + 0.9
                very_good_gap = valid3 and eta3 > ego_eta + 1.5

                if unsafe_now and speed_kmh > 5.0:
                    timing_penalty += 0.22 + 0.035 * max(ego_eta + 0.6 - eta1, 0.0)
                if unsafe_second and speed_kmh > 5.0:
                    timing_penalty += 0.14 + 0.025 * max(ego_eta + 0.3 - eta2, 0.0)

                close_to_entry = dist_to_conflict < 8.0

                # much stronger anti-freeze
                if close_to_entry and speed_kmh < 3.0 and not in_conflict:
                    hesitation_penalty += 0.03
                    if valid3 and eta3 > ego_eta + 1.0:
                        hesitation_penalty += 0.04
                    elif valid2 and eta2 > ego_eta + 0.8:
                        hesitation_penalty += 0.025

                # punish severe freezing when there is clearly room later
                if close_to_entry and speed_kmh < 1.5 and very_good_gap and not in_conflict:
                    hesitation_penalty += 0.05

                # stronger reward for actually committing in a plausible late gap
                if good_late_gap and speed_kmh > 6.0:
                    commit_bonus += 0.06

                # explicit bonus for entering / starting to enter conflict with a good window
                if in_conflict and good_late_gap and speed_kmh > 5.0:
                    entry_commit_bonus += 0.08

                if valid3 and eta3 > ego_eta + 1.4 and speed_kmh < 2.0:
                    hesitation_penalty += 0.03

        else:
            phase_bonus = 0.06 * min(speed_kmh / 18.0, 1.0)
            phase_penalty = 0.06 if speed_kmh < 5.0 and in_conflict else 0.0

        conflict_occupancy_penalty = 0.05 if in_conflict and speed_kmh < 6.0 else 0.0

        reward = (
            progress_reward
            - comfort_penalty
            - traffic_penalty
            + phase_bonus
            + yield_bonus
            + timing_bonus
            + commit_bonus
            + entry_commit_bonus
            - timing_penalty
            - hesitation_penalty
            - phase_penalty
            - conflict_occupancy_penalty
            - 0.006
        )

        if success:
            reward += 35.0
        if crashed:
            reward -= 40.0
        if timed_out:
            reward -= 20.0

        info = {
            "progress_reward": progress_reward,
            "comfort_penalty": comfort_penalty,
            "traffic_penalty": traffic_penalty,
            "phase_bonus": phase_bonus,
            "yield_bonus": yield_bonus,
            "timing_bonus": timing_bonus,
            "timing_penalty": timing_penalty,
            "phase_penalty": phase_penalty,
            "hesitation_penalty": hesitation_penalty,
            "conflict_occupancy_penalty": conflict_occupancy_penalty,
            "commit_bonus": commit_bonus,
            "goal_distance": goal_distance,
            "dist_to_conflict": dist_to_conflict,
            "forward_headway_m": forward_headway,
            "forward_ttc_s": forward_ttc,
            "decision_active": float(decision_active),
            "dist_to_decision_m": self.last_dist_to_decision_m,
            "lead_clear_time_s": self.last_lead_clear_time_s,
            "follow_arrival_time_s": self.last_follow_arrival_time_s,
            "ego_arrival_time_s": self.last_ego_arrival_time_s,
            "gap_margin_s": self.last_gap_margin_s,
            "follow_closing_speed_mps": self.last_follow_closing_speed_mps,
            "lead_length_m": self.last_lead_length_m,
            "first_pass_seen_approach": 0.0,
            "first_pass_completed": 0.0,
            "decision_gate_open": 1.0,
            "first_pass_wait_elapsed_s": 0.0,
            "upcoming_eta_1_s": self.last_upcoming_etas_s[0],
            "upcoming_eta_2_s": self.last_upcoming_etas_s[1],
            "upcoming_eta_3_s": self.last_upcoming_etas_s[2],
            "gap12_s": self.last_gap12_s,
            "gap23_s": self.last_gap23_s,
        }
        return reward, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = int(seed)

        self.collision_events.clear()
        reset_vehicle(self.world, self.handles.ego, self.handles.ego_start)
        if self.handles.traffic_stream is not None:
            sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
            self.handles.traffic_stream.reset(sim_t)
        self.world.tick()
        self._warmup_traffic()

        self.episode_start_t = self.world.get_snapshot().timestamp.elapsed_seconds
        self.prev_goal_distance = self._goal_distance()
        self.step_count = 0
        self.route_index = 0
        self.first_pass_seen_approach = False
        self.first_pass_completed = False
        self.first_pass_wait_elapsed_s = 0.0
        self.first_pass_last_signed_dist_m = None
        self.first_pass_release_no_vehicle = True
        self.last_upcoming_etas_s = [999.0, 999.0, 999.0]
        self.last_upcoming_speeds_mps = [0.0, 0.0, 0.0]
        self.last_upcoming_signed_dist_m = [999.0, 999.0, 999.0]
        self.last_gap12_s = 999.0
        self.last_gap23_s = 999.0
        self.local_trajectory = self._trajectory_points_from_action(np.zeros(self.action_space.shape[0], dtype=np.float32))
        self.last_preview_target = self._tracking_target()
        self._update_phase()
        self._draw_debug()

        return self._get_obs(), {"scenario": self.scenario_name}

    def step(self, action: np.ndarray):
        self.step_count += 1
        self.local_trajectory = self._trajectory_points_from_action(action)
        target_point = self._tracking_target()
        self.last_preview_target = target_point

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        if self.handles.traffic_stream is not None:
            self.handles.traffic_stream.tick(sim_t)

        target_wp = self._trajectory_waypoint_proxy(target_point)
        self._update_phase()
        ego_xy_pre = _xy(self.handles.ego.get_location())
        dist_to_conflict_pre = _distance_point_to_polyline(ego_xy_pre, self.conflict_center_xy) if self.conflict_center_xy else 999.0
        touch_dist_th_pre = 0.5 * self.conflict_width_m + self.ego_half_width_m
        in_conflict_pre = dist_to_conflict_pre <= touch_dist_th_pre
        decision_active_pre = self._is_decision_active(ego_xy_pre, in_conflict_pre)
        self._update_conflict_flow_cache(decision_active_pre)
        desired_speed_kmh = target_point.speed_kmh
        control = self.pid_controller.run_step(desired_speed_kmh, target_wp)
        control.steer = float(_clamp(control.steer, -0.85, 0.85))
        control.throttle = float(_clamp(control.throttle, 0.0, 0.8))
        control.brake = float(_clamp(control.brake, 0.0, 0.6))

        self.handles.ego.apply_control(control)
        self.world.tick()

        self._update_phase()
        self._draw_debug()

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        goal_distance = self._goal_distance()
        success = goal_distance <= self.dest_radius_m
        crashed = len(self.collision_events) > 0
        timed_out = (sim_t - self.episode_start_t) >= self.timeout_s
        terminated = success or crashed
        truncated = timed_out and not terminated

        reward, reward_info = self._compute_reward(success, crashed, timed_out)

        info: Dict[str, Any] = {
            "scenario": self.scenario_name,
            "phase": self.phase_name,
            "success": success,
            "crashed": crashed,
            "timed_out": timed_out,
            "goal_distance": reward_info["goal_distance"],
            "speed_kmh": _speed_kmh(self.handles.ego),
            "dist_to_conflict": reward_info["dist_to_conflict"],
            "phase_touch_idx": self.phase_touch_idx,
            "progress_reward": reward_info["progress_reward"],
            "comfort_penalty": reward_info["comfort_penalty"],
            "traffic_penalty": reward_info["traffic_penalty"],
            "phase_bonus": reward_info["phase_bonus"],
            "yield_bonus": reward_info["yield_bonus"],
            "phase_penalty": reward_info["phase_penalty"],
            "forward_headway_m": reward_info["forward_headway_m"],
            "forward_ttc_s": reward_info["forward_ttc_s"],
            "decision_active": bool(reward_info["decision_active"]),
            "dist_to_decision_m": reward_info["dist_to_decision_m"],
            "lead_clear_time_s": reward_info["lead_clear_time_s"],
            "follow_arrival_time_s": reward_info["follow_arrival_time_s"],
            "ego_arrival_time_s": reward_info["ego_arrival_time_s"],
            "gap_margin_s": reward_info["gap_margin_s"],
            "follow_closing_speed_mps": reward_info["follow_closing_speed_mps"],
            "lead_length_m": reward_info["lead_length_m"],
            "first_pass_seen_approach": bool(reward_info["first_pass_seen_approach"]),
            "first_pass_completed": bool(reward_info["first_pass_completed"]),
            "decision_gate_open": bool(reward_info["decision_gate_open"]),
            "first_pass_wait_elapsed_s": reward_info["first_pass_wait_elapsed_s"],
            "upcoming_eta_1_s": reward_info["upcoming_eta_1_s"],
            "upcoming_eta_2_s": reward_info["upcoming_eta_2_s"],
            "upcoming_eta_3_s": reward_info["upcoming_eta_3_s"],
            "gap12_s": reward_info["gap12_s"],
            "gap23_s": reward_info["gap23_s"],
        }

        return self._get_obs(), reward, terminated, truncated, info

    def close(self) -> None:
        if self.handles is None:
            return
        try:
            try:
                self.handles.collision_sensor.stop()
            except Exception:
                pass
            tm = self.client.get_trafficmanager(self.tm_port)
            tls = self.world.get_actors().filter("traffic.traffic_light*")
            if len(tls) > 0:
                tls[0].freeze(False)

            settings = self.world.get_settings()
            if hasattr(settings, "no_rendering_mode"):
                settings.no_rendering_mode = False
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            tm.set_synchronous_mode(False)
            self.world.wait_for_tick(2.0)

            if self.handles.traffic_stream is not None:
                self.handles.traffic_stream.destroy_all()

            destroy_cmds = [carla.command.DestroyActor(a.id) for a in self.handles.actors if a.is_alive]
            if destroy_cmds:
                self.client.apply_batch_sync(destroy_cmds, False)
        finally:
            self.handles = None
