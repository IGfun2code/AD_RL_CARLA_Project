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


def _vec_mag(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def _speed_kmh(vehicle: carla.Vehicle) -> float:
    vel = vehicle.get_velocity()
    return 3.6 * _vec_mag(vel.x, vel.y, vel.z)


def _yaw_deg(v: carla.Vector3D) -> float:
    return math.degrees(math.atan2(v.y, v.x))


def _normalize_angle_deg(angle_deg: float) -> float:
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


@dataclass
class EpisodeStats:
    steps: int = 0
    progress_reward: float = 0.0
    comfort_penalty: float = 0.0
    traffic_penalty: float = 0.0
    idle_penalty: float = 0.0
    total_reward: float = 0.0


class CarlaPPOEnv(gym.Env):
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
    ):
        super().__init__()
        if scenario_name not in SCENARIO_PRESETS:
            raise ValueError(f"Unknown scenario_name={scenario_name}")

        self.host = host
        self.port = port
        self.seed_value = seed
        self.scenario_name = scenario_name
        self.hide_layers = hide_layers
        self.freeze_lights = freeze_lights
        self.timeout_s = timeout_s
        self.dest_radius_m = dest_radius_m
        self.warmup_s = warmup_s
        self.obs_vehicle_count = obs_vehicle_count
        self.obs_radius_m = obs_radius_m
        self.target_speed_kmh = target_speed_kmh
        self.route_sampling_resolution = 2.0
        self.route_lookahead_choices = 6
        self.min_target_speed_kmh = 0.0
        self.max_target_speed_kmh = 45.0

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
        self.route_index = 0
        self.pid_controller: Optional[VehiclePIDController] = None
        self.last_target_speed_kmh = self.target_speed_kmh
        self.last_selected_lookahead = 0

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        obs_dim = 8 + self.obs_vehicle_count * 5
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._build_scene()

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
        self.pid_controller = VehiclePIDController(
            self.handles.ego,
            args_lateral={"K_P": 1.95, "K_I": 0.05, "K_D": 0.2, "dt": self.dt},
            args_longitudinal={"K_P": 1.0, "K_I": 0.05, "K_D": 0.0, "dt": self.dt},
            max_throttle=0.75,
            max_brake=0.5,
            max_steering=0.8,
        )
        self.global_route_planner = GlobalRoutePlanner(self.world.get_map(), self.route_sampling_resolution)
        self._refresh_route_from_current_pose(self.handles.ego_start.location)
        self.last_target_speed_kmh = self.target_speed_kmh
        self.last_selected_lookahead = 0

    def _refresh_route_from_current_pose(self, start_location: carla.Location) -> None:
        end_wp = self.world.get_map().get_waypoint(self.handles.ego_destination)
        self.route_plan = self.global_route_planner.trace_route(start_location, end_wp.transform.location)
        if not self.route_plan:
            self.route_plan = [(end_wp, None)]
        elif self.route_plan[-1][0].id != end_wp.id:
            self.route_plan.append((end_wp, self.route_plan[-1][1]))
        self.route_waypoints = [wp for wp, _ in self.route_plan]
        self.route_index = 0

    def _advance_route_index(self) -> None:
        if not self.route_waypoints:
            self.route_index = 0
            return
        ego_loc = self.handles.ego.get_location()
        upper = min(len(self.route_waypoints), self.route_index + 25)
        best_idx = self.route_index
        best_dist = float("inf")
        for idx in range(self.route_index, upper):
            dist = ego_loc.distance(self.route_waypoints[idx].transform.location)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        self.route_index = best_idx

    def _select_target_waypoint(self, action_value: float) -> carla.Waypoint:
        self._refresh_route_from_current_pose(self.handles.ego.get_location())
        lookahead = int(
            np.clip(
                round(((action_value + 1.0) * 0.5) * (self.route_lookahead_choices - 1)),
                0,
                self.route_lookahead_choices - 1,
            )
        )
        self.last_selected_lookahead = lookahead
        target_idx = min(len(self.route_waypoints) - 1, 1 + lookahead)
        return self.route_waypoints[target_idx]

    def _target_speed_from_action(self, action_value: float) -> float:
        speed = self.min_target_speed_kmh + ((action_value + 1.0) * 0.5) * (
            self.max_target_speed_kmh - self.min_target_speed_kmh
        )
        self.last_target_speed_kmh = float(np.clip(speed, self.min_target_speed_kmh, self.max_target_speed_kmh))
        return self.last_target_speed_kmh

    def _active_traffic(self) -> List[carla.Vehicle]:
        if self.handles is None or self.handles.traffic_stream is None:
            return []
        return [v for v in self.handles.traffic_stream.vehicles if v.is_alive]

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

    def _goal_distance(self) -> float:
        return self.handles.ego.get_location().distance(self.handles.ego_destination)

    def _local_relative_features(self, other: carla.Vehicle) -> Tuple[float, float, float, float, float]:
        ego_tf = self.handles.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        dx = other.get_location().x - ego_loc.x
        dy = other.get_location().y - ego_loc.y
        cos_yaw = math.cos(-ego_yaw)
        sin_yaw = math.sin(-ego_yaw)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        ego_v = self.handles.ego.get_velocity()
        other_v = other.get_velocity()
        rel_vx = other_v.x - ego_v.x
        rel_vy = other_v.y - ego_v.y
        dist = math.sqrt(dx * dx + dy * dy)

        return (
            float(np.clip(local_x / self.obs_radius_m, -1.0, 1.0)),
            float(np.clip(local_y / self.obs_radius_m, -1.0, 1.0)),
            float(np.clip(rel_vx / 20.0, -1.0, 1.0)),
            float(np.clip(rel_vy / 20.0, -1.0, 1.0)),
            float(np.clip(dist / self.obs_radius_m, 0.0, 1.0)),
        )

    def _get_obs(self) -> np.ndarray:
        ego = self.handles.ego
        goal_distance = self._goal_distance()
        speed = _speed_kmh(ego)
        control = ego.get_control()

        to_goal = self.handles.ego_destination - ego.get_location()
        heading = ego.get_transform().get_forward_vector()
        heading_deg = _yaw_deg(heading)
        goal_heading_deg = _yaw_deg(to_goal)
        heading_error_deg = _normalize_angle_deg(goal_heading_deg - heading_deg)
        heading_error_rad = math.radians(heading_error_deg)

        obs: List[float] = [
            float(np.clip(speed / max(self.target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(goal_distance / 120.0, 0.0, 1.0)),
            float(math.sin(heading_error_rad)),
            float(math.cos(heading_error_rad)),
            float(np.clip(control.steer, -1.0, 1.0)),
            float(np.clip(self.last_target_speed_kmh / max(self.max_target_speed_kmh, 1.0), 0.0, 1.0)),
            float(np.clip(self.last_selected_lookahead / max(self.route_lookahead_choices - 1, 1), 0.0, 1.0)),
            float(np.clip(self.route_index / max(len(self.route_waypoints) - 1, 1), 0.0, 1.0)),
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
            obs.extend(self._local_relative_features(vehicle))

        while len(obs) < self.observation_space.shape[0]:
            obs.extend([0.0] * 5)

        return np.asarray(obs, dtype=np.float32)

    def _compute_reward(self, success: bool, crashed: bool, timed_out: bool) -> Tuple[float, Dict[str, float]]:
        goal_distance = self._goal_distance()
        progress = 0.0
        if self.prev_goal_distance is not None:
            progress = self.prev_goal_distance - goal_distance
        self.prev_goal_distance = goal_distance

        control = self.handles.ego.get_control()
        speed_kmh = _speed_kmh(self.handles.ego)
        speed_error = abs(speed_kmh - self.target_speed_kmh) / max(self.target_speed_kmh, 1.0)
        comfort_penalty = 0.02 * abs(control.steer) + 0.015 * control.brake + 0.015 * speed_error

        traffic_penalty = 0.0
        ego_loc = self.handles.ego.get_location()
        for vehicle in self._active_traffic():
            dist = ego_loc.distance(vehicle.get_location())
            if dist < 8.0:
                traffic_penalty += (8.0 - dist) * 0.05

        idle_penalty = 0.0
        if speed_kmh < 5.0 and goal_distance > 12.0:
            idle_penalty = 0.08

        heading_penalty = 0.0
        to_goal = self.handles.ego_destination - self.handles.ego.get_location()
        heading = self.handles.ego.get_transform().get_forward_vector()
        heading_error_deg = _normalize_angle_deg(_yaw_deg(to_goal) - _yaw_deg(heading))
        if goal_distance > 8.0:
            heading_penalty = 0.02 * abs(math.sin(math.radians(heading_error_deg)))

        reward = 0.8 * progress - comfort_penalty - traffic_penalty - idle_penalty - heading_penalty - 0.01
        if success:
            reward += 30.0
        if crashed:
            reward -= 30.0
        if timed_out:
            reward -= 15.0

        info = {
            "progress_reward": 0.8 * progress,
            "comfort_penalty": comfort_penalty,
            "traffic_penalty": traffic_penalty,
            "idle_penalty": idle_penalty,
            "goal_distance": goal_distance,
        }
        return reward, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed_value = seed

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
        obs = self._get_obs()
        return obs, {"scenario": self.scenario_name}

    def step(self, action: np.ndarray):
        target_waypoint = self._select_target_waypoint(float(np.clip(action[0], -1.0, 1.0)))
        target_speed_kmh = self._target_speed_from_action(float(np.clip(action[1], -1.0, 1.0)))

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        if self.handles.traffic_stream is not None:
            self.handles.traffic_stream.tick(sim_t)

        control = self.pid_controller.run_step(target_speed_kmh, target_waypoint)
        self.handles.ego.apply_control(control)
        self.world.tick()

        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        goal_distance = self._goal_distance()
        speed_kmh = _speed_kmh(self.handles.ego)
        success = goal_distance <= self.dest_radius_m
        crashed = len(self.collision_events) > 0
        timed_out = (sim_t - self.episode_start_t) >= self.timeout_s
        made_progress = (
            self.prev_goal_distance is None or
            (self.prev_goal_distance - goal_distance) > 0.02 or
            speed_kmh > 5.0
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
        self.stats.idle_penalty += reward_info["idle_penalty"]
        if stuck:
            reward -= 8.0
        self.stats.total_reward += reward

        info: Dict[str, Any] = {
            "scenario": self.scenario_name,
            "success": success,
            "crashed": crashed,
            "timed_out": timed_out,
            "stuck": stuck,
            "goal_distance": reward_info["goal_distance"],
            "speed_kmh": speed_kmh,
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
