from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import carla

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym  # type: ignore
    from gym import spaces  # type: ignore

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scenarios.scene import build_scene, reset_vehicle
from scenarios.scenario_catalog import SCENARIO_PRESETS


def ensure_agents_importable() -> None:
    try:
        import agents  # noqa: F401
        return
    except Exception:
        pass

    carla_root = os.environ.get("CARLA_ROOT", "")
    if carla_root:
        candidate = os.path.join(carla_root, "PythonAPI", "carla")
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.append(candidate)

    import agents  # noqa: F401


ensure_agents_importable()
from agents.navigation.controller import VehiclePIDController  # type: ignore
from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore


# ------------------------------
# Small geometry helpers
# ------------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _vec_mag(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _speed_kmh(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6 * _vec_mag(vel.x, vel.y, vel.z)


def _carla_loc_to_np(loc: carla.Location) -> np.ndarray:
    return np.array([float(loc.x), float(loc.y)], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _heading_from_rotation_deg(rot: carla.Rotation) -> float:
    return math.radians(float(rot.yaw))


def _unit_from_heading(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def _normal_from_heading(theta: float) -> np.ndarray:
    return np.array([-math.sin(theta), math.cos(theta)], dtype=np.float32)


# ------------------------------
# Simple proxy waypoint for CARLA PID
# ------------------------------
class SimpleWaypoint:
    """Minimal object that mimics the CARLA waypoint interface used by VehiclePIDController."""

    def __init__(self, x: float, y: float, yaw_rad: float, z: float = 0.0):
        self.transform = carla.Transform(
            carla.Location(x=float(x), y=float(y), z=float(z)),
            carla.Rotation(pitch=0.0, yaw=math.degrees(yaw_rad), roll=0.0),
        )


@dataclass
class EpisodeStats:
    steps: int = 0
    progress_reward: float = 0.0
    comfort_penalty: float = 0.0
    traffic_penalty: float = 0.0
    lane_penalty: float = 0.0
    idle_penalty: float = 0.0
    total_reward: float = 0.0


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw_rad: float
    speed_kmh: float

    def as_np(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float32)


class CarlaPPOEnv(gym.Env):
    """
    Global planner: CARLA GlobalRoutePlanner (A*) generates the route backbone.
    Local planner: PPO generates a local trajectory segment every simulator step.
    Tracking: CARLA VehiclePIDController tracks a preview waypoint from the PPO-generated trajectory.

    PPO action = [lat_1, lat_2, lat_3, reconnect_heading_bias, target_speed]
    The trajectory is represented internally as waypoints to keep PPO's action space compact.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        scenario_name: str,
        host: str = "127.0.0.1",
        port: int = 2000,
        seed: int = 0,
        hide_layers: bool = True,
        freeze_lights: bool = True,
        timeout_s: float = 60.0,
        dest_radius_m: float = 5.0,
        warmup_s: float = 3.0,
        obs_vehicle_count: int = 4,
        obs_radius_m: float = 60.0,
        target_speed_kmh: float = 35.0,
        debug_draw_trajectory: bool = False,
        debug_draw_interval_steps: int = 10,
        debug_draw_lifetime_s: float = 1.0,
        debug_draw_route: bool = True,
    ):
        super().__init__()
        if scenario_name not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario_name={scenario_name}")

        self.host = host
        self.port = int(port)
        self.seed_value = int(seed)
        self.scenario_name = scenario_name
        self.hide_layers = hide_layers
        self.freeze_lights = freeze_lights
        self.timeout_s = float(timeout_s)
        self.dest_radius_m = float(dest_radius_m)
        self.warmup_s = float(warmup_s)
        self.obs_vehicle_count = int(obs_vehicle_count)
        self.obs_radius_m = float(obs_radius_m)
        self.target_speed_kmh = float(target_speed_kmh)

        self.debug_draw_trajectory = bool(debug_draw_trajectory)
        self.debug_draw_interval_steps = max(1, int(debug_draw_interval_steps))
        self.debug_draw_lifetime_s = float(debug_draw_lifetime_s)
        self.debug_draw_route = bool(debug_draw_route)

        # Planning/tracking configuration.
        self.route_sampling_resolution = 2.0
        self.local_anchor_distances_m = [8.0, 18.0, 30.0]
        self.max_lateral_offset_m = 2.0
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
        self.prev_goal_distance: Optional[float] = None
        self.episode_start_t: float = 0.0
        self.stats = EpisodeStats()
        self.stagnation_steps = 0

        self.route_plan: List[Any] = []
        self.route_waypoints: List[carla.Waypoint] = []
        self.route_points_xy: List[np.ndarray] = []
        self.route_index = 0
        self.local_trajectory: List[TrajectoryPoint] = []
        self.pid_controller: Optional[VehiclePIDController] = None
        self.last_target_speed_kmh = self.target_speed_kmh
        self.last_lateral_offsets_m = [0.0] * len(self.local_anchor_distances_m)
        self.last_heading_bias_deg = 0.0
        self.last_preview_target: Optional[TrajectoryPoint] = None
        self.prev_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

        # 3 lateral offsets + 1 reconnect heading bias + 1 speed command.
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Ego/route/tracking summary: 12 dims.
        # For each of up to N nearby vehicles: 8 dims = rel_x, rel_y, rel_vx, rel_vy, dist, length, width, safety_radius.
        obs_dim = 12 + self.obs_vehicle_count * 8
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._build_scene()

    # ------------------------------
    # Scene / reset
    # ------------------------------
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
            hide_layers=self.hide_layers,
            freeze_lights=self.freeze_lights,
            use_stream=True,
            mean_headway_s=preset["mean_headway"],
            burst_prob=preset["burst_prob"],
        )
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        self.dt = settings.fixed_delta_seconds or 0.05
        self.collision_events = []
        self.handles.collision_sensor.listen(lambda event: self.collision_events.append(event))
        self.world.tick()
        self._reset_guidance_stack()

    def _reset_guidance_stack(self) -> None:
        self.global_route_planner = GlobalRoutePlanner(self.world.get_map(), self.route_sampling_resolution)
        self.pid_controller = VehiclePIDController(
            self.handles.ego,
            args_lateral={"K_P": 1.35, "K_I": 0.00, "K_D": 0.25, "dt": self.dt},
            args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": self.dt},
            max_throttle=0.75,
            max_brake=0.5,
            max_steering=0.8,
        )
        self.prev_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)
        self._refresh_route_from_current_pose(self.handles.ego.get_location())
        self.last_target_speed_kmh = self.target_speed_kmh
        self.last_lateral_offsets_m = [0.0] * len(self.local_anchor_distances_m)
        self.last_heading_bias_deg = 0.0
        self.last_preview_target = None
        self.local_trajectory = []

    def _warmup_traffic(self) -> None:
        if self.handles is None or self.handles.traffic_stream is None:
            return
        start_t = self.world.get_snapshot().timestamp.elapsed_seconds
        deadline_t = start_t + max(self.warmup_s + 10.0, 15.0)
        while True:
            sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
            self.handles.traffic_stream.tick(sim_t)
            self.world.tick()
            enough_time = (sim_t - start_t) >= self.warmup_s
            enough_traffic = len(self._active_traffic()) >= min(3, self.obs_vehicle_count)
            if (enough_time and enough_traffic) or sim_t >= deadline_t:
                return

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
        self._reset_guidance_stack()

        self.episode_start_t = self.world.get_snapshot().timestamp.elapsed_seconds
        self.prev_goal_distance = self._goal_distance()
        self.stats = EpisodeStats()
        self.stagnation_steps = 0
        self.local_trajectory = self._trajectory_points_from_action(np.zeros(self.action_space.shape[0], dtype=np.float32))
        self.last_preview_target = self._tracking_target()
        if self.debug_draw_trajectory:
            self._draw_debug_plans(force=True)
        obs = self._get_obs()
        return obs, {"scenario": self.scenario_name}

    # ------------------------------
    # Route / trajectory generation
    # ------------------------------
    def _goal_distance(self) -> float:
        return self.handles.ego.get_location().distance(self.handles.ego_destination)

    def _refresh_route_from_current_pose(self, start_location: carla.Location) -> None:
        end_wp = self.world.get_map().get_waypoint(self.handles.ego_destination)
        self.route_plan = self.global_route_planner.trace_route(start_location, end_wp.transform.location)
        if not self.route_plan:
            self.route_plan = [(end_wp, None)]
        elif self.route_plan[-1][0].id != end_wp.id:
            self.route_plan.append((end_wp, self.route_plan[-1][1]))
        self.route_waypoints = [wp for wp, _ in self.route_plan]
        self.route_points_xy = [_carla_loc_to_np(wp.transform.location) for wp in self.route_waypoints]
        self.route_index = 0

    def _advance_route_index(self) -> None:
        if not self.route_points_xy:
            self.route_index = 0
            return
        ego_xy = _carla_loc_to_np(self.handles.ego.get_location())
        upper = min(len(self.route_points_xy), self.route_index + 30)
        best_idx = self.route_index
        best_dist = float("inf")
        for idx in range(self.route_index, upper):
            dist = _dist(ego_xy, self.route_points_xy[idx])
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        self.route_index = best_idx

    def _route_point_at_distance(self, start_idx: int, target_dist_m: float) -> Tuple[np.ndarray, float, int]:
        if not self.route_points_xy:
            ego_xy = _carla_loc_to_np(self.handles.ego.get_location())
            ego_heading = _heading_from_rotation_deg(self.handles.ego.get_transform().rotation)
            return ego_xy, ego_heading, 0

        acc = 0.0
        idx = max(0, min(start_idx, len(self.route_points_xy) - 1))
        prev = self.route_points_xy[idx]
        while idx + 1 < len(self.route_points_xy) and acc < target_dist_m:
            nxt = self.route_points_xy[idx + 1]
            seg = _dist(prev, nxt)
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
            heading = math.atan2(float((p1 - p0)[1]), float((p1 - p0)[0])) if _dist(p0, p1) > 1e-6 else 0.0
        else:
            heading = 0.0
        return self.route_points_xy[idx].astype(np.float32), heading, idx

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

    def _catmull_rom_chain(self, pts: Sequence[np.ndarray], samples_per_seg: int = 8) -> List[np.ndarray]:
        if len(pts) < 2:
            return [pts[0]] if pts else []
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
        self._refresh_route_from_current_pose(self.handles.ego.get_location())
        self._advance_route_index()

        ego_tf = self.handles.ego.get_transform()
        ego_xy = _carla_loc_to_np(ego_tf.location)
        ego_heading = _heading_from_rotation_deg(ego_tf.rotation)

        control_pts: List[np.ndarray] = [ego_xy]
        reconnect_idx = self.route_index
        for d_m, lat_m in zip(self.local_anchor_distances_m, lateral_offsets_m):
            center_pt, tangent_heading, reconnect_idx = self._route_point_at_distance(self.route_index, d_m)
            normal = _normal_from_heading(tangent_heading)
            offset_pt = center_pt + normal * lat_m
            control_pts.append(offset_pt.astype(np.float32))

        reconnect_pt, reconnect_heading, reconnect_idx = self._route_point_at_distance(
            self.route_index, min(self.max_preview_distance_m, self.local_anchor_distances_m[-1] + 10.0)
        )
        reconnect_heading += math.radians(heading_bias_deg)
        reconnect_pt = reconnect_pt + _normal_from_heading(reconnect_heading) * lateral_offsets_m[-1]
        control_pts.append(reconnect_pt.astype(np.float32))

        spline_pts = self._catmull_rom_chain(control_pts, samples_per_seg=8)
        if len(spline_pts) < 2:
            spline_pts = control_pts

        dense_pts: List[np.ndarray] = [spline_pts[0]]
        for p in spline_pts[1:]:
            seg_len = _dist(dense_pts[-1], p)
            if seg_len < 0.5:
                continue
            n_insert = max(1, int(seg_len / self.local_traj_point_spacing_m))
            start = dense_pts[-1]
            for k in range(1, n_insert + 1):
                alpha = k / float(n_insert)
                dense_pts.append((start + alpha * (p - start)).astype(np.float32))

        trajectory: List[TrajectoryPoint] = []
        if len(dense_pts) == 1:
            dense_pts.append(dense_pts[0] + _unit_from_heading(ego_heading) * 0.5)

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

        # Append remaining global route so the full path still terminates at the destination.
        for idx in range(min(reconnect_idx + 1, len(self.route_points_xy) - 1), len(self.route_points_xy)):
            pt = self.route_points_xy[idx]
            if trajectory and _dist(trajectory[-1].as_np(), pt) < 1.0:
                continue
            if idx > 0:
                prev = self.route_points_xy[idx - 1]
                yaw = math.atan2(float((pt - prev)[1]), float((pt - prev)[0])) if _dist(prev, pt) > 1e-6 else trajectory[-1].yaw_rad
            else:
                yaw = trajectory[-1].yaw_rad if trajectory else ego_heading
            trajectory.append(TrajectoryPoint(float(pt[0]), float(pt[1]), yaw, target_speed_kmh))

        if not trajectory:
            trajectory = [TrajectoryPoint(float(ego_xy[0]), float(ego_xy[1]), ego_heading, target_speed_kmh)]
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
            d = _dist(ego_xy, pt)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        return best_idx

    def _tracking_target(self) -> TrajectoryPoint:
        ego_xy = _carla_loc_to_np(self.handles.ego.get_location())
        if not self.local_trajectory:
            ego_heading = _heading_from_rotation_deg(self.handles.ego.get_transform().rotation)
            return TrajectoryPoint(float(ego_xy[0]), float(ego_xy[1]), ego_heading, self.last_target_speed_kmh)

        nearest_idx = self._nearest_trajectory_index(ego_xy)
        accum = 0.0
        for idx in range(nearest_idx, len(self.local_trajectory) - 1):
            p0 = self.local_trajectory[idx].as_np()
            p1 = self.local_trajectory[idx + 1].as_np()
            accum += _dist(p0, p1)
            if accum >= self.track_preview_m:
                return self.local_trajectory[idx + 1]
        return self.local_trajectory[-1]

    def _trajectory_waypoint_proxy(self, target: TrajectoryPoint) -> SimpleWaypoint:
        ego_z = self.handles.ego.get_location().z
        return SimpleWaypoint(target.x, target.y, target.yaw_rad, z=ego_z)

    # ------------------------------
    # Observation / reward
    # ------------------------------
    def _active_traffic(self) -> List[carla.Vehicle]:
        if self.handles is None or self.handles.traffic_stream is None:
            return []
        return [v for v in self.handles.traffic_stream.vehicles if v.is_alive]

    def _vehicle_half_extents(self, vehicle: carla.Vehicle) -> Tuple[float, float]:
        bb = vehicle.bounding_box.extent
        return float(bb.x), float(bb.y)

    def _neighbor_features(self, vehicle: carla.Vehicle) -> Tuple[float, ...]:
        ego_tf = self.handles.ego.get_transform()
        ego_xy = _carla_loc_to_np(ego_tf.location)
        ego_heading = _heading_from_rotation_deg(ego_tf.rotation)

        veh_xy = _carla_loc_to_np(vehicle.get_location())
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

    def _lane_offset_metric(self) -> float:
        ego_loc = self.handles.ego.get_location()
        wp = self.world.get_map().get_waypoint(
            ego_loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        return float(ego_loc.distance(wp.transform.location))

    def _get_obs(self) -> np.ndarray:
        ego = self.handles.ego
        goal_distance = self._goal_distance()
        speed = _speed_kmh(ego)
        control = ego.get_control()
        self._advance_route_index()

        ego_tf = ego.get_transform()
        ego_xy = _carla_loc_to_np(ego_tf.location)
        ego_heading = _heading_from_rotation_deg(ego_tf.rotation)

        if self.route_points_xy:
            route_idx = min(self.route_index, len(self.route_points_xy) - 1)
            ref_pt = self.route_points_xy[route_idx]
            route_heading = ego_heading
            if route_idx + 1 < len(self.route_points_xy):
                nxt = self.route_points_xy[route_idx + 1]
                if _dist(ref_pt, nxt) > 1e-6:
                    route_heading = math.atan2(float((nxt - ref_pt)[1]), float((nxt - ref_pt)[0]))
            route_error = _wrap_pi(route_heading - ego_heading)
            route_delta = ref_pt - ego_xy
            lane_signed = float(np.dot(route_delta, _normal_from_heading(route_heading)))
        else:
            route_error = 0.0
            lane_signed = 0.0

        track_target = self._tracking_target()
        track_delta = track_target.as_np() - ego_xy
        track_dist = float(np.linalg.norm(track_delta))
        track_heading_error = _wrap_pi(track_target.yaw_rad - ego_heading)

        obs: List[float] = [
            float(np.clip(speed / max(self.max_target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(goal_distance / 120.0, 0.0, 1.0)),
            float(np.sin(route_error)),
            float(np.cos(route_error)),
            float(np.clip(lane_signed / 6.0, -1.0, 1.0)),
            float(np.sin(track_heading_error)),
            float(np.cos(track_heading_error)),
            float(np.clip(track_dist / self.max_preview_distance_m, 0.0, 1.0)),
            float(np.clip(control.steer, -1.0, 1.0)),
            float(np.clip(self.last_target_speed_kmh / max(self.max_target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(np.mean(self.last_lateral_offsets_m) / max(self.max_lateral_offset_m, 1e-3), -1.0, 1.0)),
            float(np.clip(self.last_heading_bias_deg / max(self.max_heading_bias_deg, 1e-3), -1.0, 1.0)),
        ]

        others: List[Tuple[float, carla.Vehicle]] = []
        for vehicle in self._active_traffic():
            if vehicle.id == ego.id:
                continue
            dist = ego.get_location().distance(vehicle.get_location())
            if dist <= self.obs_radius_m:
                others.append((dist, vehicle))
        others.sort(key=lambda item: item[0])

        for _, vehicle in others[: self.obs_vehicle_count]:
            obs.extend(self._neighbor_features(vehicle))

        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0] * 8)

        return np.asarray(obs, dtype=np.float32)

    def _compute_reward(self, success: bool, crashed: bool, timed_out: bool) -> Tuple[float, Dict[str, float]]:
        goal_distance = self._goal_distance()
        progress = 0.0
        if self.prev_goal_distance is not None:
            progress = self.prev_goal_distance - goal_distance
        self.prev_goal_distance = goal_distance

        control = self.handles.ego.get_control()
        speed_kmh = _speed_kmh(self.handles.ego)
        speed_error = abs(speed_kmh - self.last_target_speed_kmh) / max(self.max_target_speed_kmh, 1.0)
        steer_rate = abs(control.steer - self.prev_control.steer)
        comfort_penalty = 0.03 * abs(control.steer) + 0.02 * control.brake + 0.015 * speed_error + 0.04 * steer_rate

        traffic_penalty = 0.0
        ego_loc = self.handles.ego.get_location()
        ego_half_len, ego_half_wid = self._vehicle_half_extents(self.handles.ego)
        ego_radius = math.sqrt(ego_half_len * ego_half_len + ego_half_wid * ego_half_wid)
        for vehicle in self._active_traffic():
            other_half_len, other_half_wid = self._vehicle_half_extents(vehicle)
            safe_gap = 1.5 + ego_radius + math.sqrt(other_half_len * other_half_len + other_half_wid * other_half_wid)
            dist = ego_loc.distance(vehicle.get_location())
            if dist < safe_gap:
                traffic_penalty += (safe_gap - dist) * 0.10

        lane_offset = self._lane_offset_metric()
        lane_penalty = max(0.0, lane_offset - 0.75) * 0.45

        idle_penalty = 0.0
        if speed_kmh < 5.0 and goal_distance > 12.0:
            idle_penalty = 0.08

        #traj_shape_penalty = 0.02 * np.mean(np.abs(self.last_lateral_offsets_m)) + 0.01 * abs(self.last_heading_bias_deg) / max(self.max_heading_bias_deg, 1e-3)
        reward = 1.0 * progress - comfort_penalty - traffic_penalty - lane_penalty - idle_penalty - 0.01
        if success:
            reward += 35.0
        if crashed:
            reward -= 40.0
        if timed_out:
            reward -= 15.0

        info = {
            "progress_reward": 1.0 * progress,
            "comfort_penalty": comfort_penalty,
            "traffic_penalty": traffic_penalty,
            "lane_penalty": lane_penalty,
            "idle_penalty": idle_penalty,
            "goal_distance": goal_distance,
        }
        return reward, info

    # ------------------------------
    # Debug visualization
    # ------------------------------
    def _draw_debug_plans(self, force: bool = False) -> None:
        if not self.debug_draw_trajectory or self.world is None:
            return
        if (not force) and (self.stats.steps % self.debug_draw_interval_steps != 0):
            return

        dbg = self.world.debug
        z0 = self.handles.ego.get_location().z + 0.35
        life = self.debug_draw_lifetime_s

        # Draw PPO local trajectory in blue.
        for i in range(len(self.local_trajectory) - 1):
            p0 = self.local_trajectory[i]
            p1 = self.local_trajectory[i + 1]
            dbg.draw_line(
                carla.Location(x=p0.x, y=p0.y, z=z0),
                carla.Location(x=p1.x, y=p1.y, z=z0),
                thickness=0.10,
                color=carla.Color(0, 0, 255),
                life_time=life,
                persistent_lines=False,
            )

        # Draw preview waypoint in orange.
        if self.last_preview_target is not None:
            t = self.last_preview_target
            dbg.draw_point(
                carla.Location(x=t.x, y=t.y, z=z0 + 0.08),
                size=0.12,
                color=carla.Color(255, 165, 0),
                life_time=life,
                persistent_lines=False,
            )

        # Draw global A* route in green if requested.
        if self.debug_draw_route and len(self.route_points_xy) >= 2:
            for i in range(len(self.route_points_xy) - 1):
                p0 = self.route_points_xy[i]
                p1 = self.route_points_xy[i + 1]
                dbg.draw_line(
                    carla.Location(x=float(p0[0]), y=float(p0[1]), z=z0 - 0.15),
                    carla.Location(x=float(p1[0]), y=float(p1[1]), z=z0 - 0.15),
                    thickness=0.05,
                    color=carla.Color(0, 255, 0),
                    life_time=life,
                    persistent_lines=False,
                )

    # ------------------------------
    # Step / close
    # ------------------------------
    def step(self, action: np.ndarray):
        self.local_trajectory = self._trajectory_points_from_action(action)
        target_point = self._tracking_target()
        self.last_preview_target = target_point

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        if self.handles.traffic_stream is not None:
            self.handles.traffic_stream.tick(sim_t)

        # Use CARLA PID to track the preview point from the PPO-generated trajectory.
        target_wp = self._trajectory_waypoint_proxy(target_point)
        control = self.pid_controller.run_step(target_point.speed_kmh, target_wp)
        control.steer = float(_clamp(control.steer, -0.85, 0.85))
        control.throttle = float(_clamp(control.throttle, 0.0, 0.8))
        control.brake = float(_clamp(control.brake, 0.0, 0.6))
        self.handles.ego.apply_control(control)
        self.world.tick()

        # Draw after the world tick so the lines are visible in the current frame.
        self._draw_debug_plans(force=False)

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        goal_distance = self._goal_distance()
        speed_kmh = _speed_kmh(self.handles.ego)
        success = goal_distance <= self.dest_radius_m
        crashed = len(self.collision_events) > 0
        timed_out = (sim_t - self.episode_start_t) >= self.timeout_s
        made_progress = (
            self.prev_goal_distance is None
            or (self.prev_goal_distance - goal_distance) > 0.02
            or speed_kmh > 5.0
        )
        if made_progress:
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        stuck = self.stagnation_steps >= 120
        terminated = success or crashed
        truncated = (timed_out or stuck) and not terminated

        reward, reward_info = self._compute_reward(success, crashed, timed_out)
        self.stats.steps += 1
        self.stats.progress_reward += reward_info["progress_reward"]
        self.stats.comfort_penalty += reward_info["comfort_penalty"]
        self.stats.traffic_penalty += reward_info["traffic_penalty"]
        self.stats.lane_penalty += reward_info["lane_penalty"]
        self.stats.idle_penalty += reward_info["idle_penalty"]
        if stuck:
            reward -= 8.0
        self.stats.total_reward += reward
        self.prev_control = control

        info: Dict[str, Any] = {
            "scenario": self.scenario_name,
            "success": success,
            "crashed": crashed,
            "timed_out": timed_out,
            "stuck": stuck,
            "goal_distance": reward_info["goal_distance"],
            "speed_kmh": speed_kmh,
            "lane_offset_m": self._lane_offset_metric(),
            "target_speed_kmh": self.last_target_speed_kmh,
            "trajectory_points": len(self.local_trajectory),
        }
        if terminated or truncated:
            info["episode"] = {
                "r": self.stats.total_reward,
                "l": self.stats.steps,
                "success": success,
                "crashed": crashed,
                "timed_out": timed_out,
                "stuck": stuck,
                "progress_reward": self.stats.progress_reward,
                "comfort_penalty": self.stats.comfort_penalty,
                "traffic_penalty": self.stats.traffic_penalty,
                "lane_penalty": self.stats.lane_penalty,
                "idle_penalty": self.stats.idle_penalty,
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
            tm = self.client.get_trafficmanager(8000)
            tls = self.world.get_actors().filter("traffic.traffic_light*")
            if len(tls) > 0:
                tls[0].freeze(False)

            settings = self.world.get_settings()
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
