#!/usr/bin/env python3
"""
Unprotected Left Turn scene setup for CARLA 0.9.16.

- Use *_Opt maps so we can hide buildings/foliage/extra visuals via MapLayer.
- Optionally freeze all traffic lights to GREEN (prevents protected phases).
- Spawn ego + dense oncoming traffic stream pinned to a path via Traffic Manager.
"""

import os
import sys
import time
import random
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict

import carla


# ---------- Optional: make sure agents/ are importable if you need them later ----------
# (Your runner will likely use BehaviorAgent/BasicAgent. Some installs include agents in the wheel,
# others require adding CARLA_ROOT/PythonAPI/carla to sys.path.)
def ensure_agents_importable():
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

def build_greedy_waypoint_path_locs(
    carla_map: carla.Map,
    start_loc: carla.Location,
    goal_loc: carla.Location,
    *,
    step_m: float = 8.0,
    sample_every: int = 2,
    max_steps: int = 800,
    goal_tol_m: float = 6.0,
) -> List[carla.Location]:
    """
    Build a list of Locations for TM.set_path() by walking waypoints from start toward goal.
    Works well for "go straight" routes because next() mostly returns a single continuation.
    """
    wp = carla_map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    path: List[carla.Location] = []
    goal = goal_loc

    for k in range(max_steps):
        if wp.transform.location.distance(goal) < goal_tol_m:
            break

        next_wps = wp.next(step_m)
        if not next_wps:
            break

        # Choose the continuation that gets us closest to the goal (greedy)
        wp = min(next_wps, key=lambda w: w.transform.location.distance(goal))

        # Downsample points (TM doesn’t need every 1m)
        if (k % sample_every) == 0:
            path.append(wp.transform.location)

    # Ensure we give TM at least 2 points
    if len(path) < 2:
        # add one point ahead and goal-ish point
        wp2 = carla_map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving).next(30.0)
        if wp2:
            path.append(wp2[0].transform.location)
        path.append(goal)

    return path


class TrafficStream:
    """
    Continuously spawns vehicles behind an anchor waypoint and despawns them near a goal location.

    Creates variable density by sampling random headways (exponential + occasional bursts).
    """
    def __init__(
        self,
        client: carla.Client,
        world: carla.World,
        tm: carla.TrafficManager,
        spawn_transform: carla.Transform,
        path_locs: List[carla.Location],
        despawn_loc: carla.Location,
        *,
        max_active: int = 30,
        mean_headway_s: float = 2.5,
        burst_prob: float = 0.15,
        spacing_m: float = 10.0,
        aggressiveness: str = "aggressive",
        despawn_radius_m: float = 15.0,
    ):
        self.client = client
        self.world = world
        self.tm = tm
        self.path_locs = path_locs
        self.despawn_loc = despawn_loc
        self.max_active = max_active
        self.mean_headway_s = mean_headway_s
        self.burst_prob = burst_prob
        self.spacing_m = spacing_m
        self.despawn_radius_m = despawn_radius_m
        self.spawn_transform = spawn_transform
        self.spawn_clearance_m = 15.0          # tweakable

        self.vehicles: List[carla.Vehicle] = []
        self.next_spawn_t = 0.0

        if aggressiveness == "aggressive":
            self.speed_diff_const = -5.0
            self.follow_dist_rng = (2.0, 2.5)
        elif aggressiveness == "normal":
            self.speed_diff_const = 0.0
            self.follow_dist_rng = (2.5, 4.0)
        else:  # cautious
            self.speed_diff_const = 10.0
            self.follow_dist_rng = (3.0, 6.0)

        self.vehicle_bps = [bp for bp in self.world.get_blueprint_library().filter("vehicle.*") if is_allowed_vehicle_bp(bp)]
    
    def reset(self, sim_t: float):
        """Clear all traffic vehicles and restart the arrival process."""
        self.destroy_all()
        self.next_spawn_t = sim_t

    def _sample_headway(self) -> float:
        # Exponential headway (Poisson arrivals)
        u = max(1e-6, random.random())
        dt = -self.mean_headway_s * math.log(u)

        # Occasional bursts: temporarily higher flow
        if random.random() < self.burst_prob:
            dt *= 0.35
        return dt

    def _spawn_clear(self, loc: carla.Location, clearance_m: float) -> bool:
        for v in self.vehicles:
            if v.is_alive and v.get_location().distance(loc) < clearance_m:
                return False
        return True

    def _try_spawn_one(self) -> Optional[carla.Vehicle]:
        spawn_t = carla.Transform(self.spawn_transform.location, self.spawn_transform.rotation)
        spawn_t.location.z += 0.5

        # Don’t spawn if entrance is “occupied”
        if not self._spawn_clear(spawn_t.location, self.spawn_clearance_m):
            return None

        bp = random.choice(self.vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))

        v = self.world.try_spawn_actor(bp, spawn_t)
        if v is None:
            return None

        v = v  # type: ignore
        v.set_autopilot(True, self.tm.get_port())
        self.tm.set_path(v, self.path_locs)

        # Keep it stable on 1-lane roads:
        follow_dist = random.uniform(*self.follow_dist_rng)
        self.tm.vehicle_percentage_speed_difference(v, self.speed_diff_const)
        self.tm.distance_to_leading_vehicle(v, follow_dist)
        self.tm.auto_lane_change(v, False)

        return v

    def _despawn_passed(self):
        keep: List[carla.Vehicle] = []
        destroy_cmds: List[carla.command.Command] = []

        for v in self.vehicles:
            if not v.is_alive:
                continue

            if v.get_location().distance(self.despawn_loc) < self.despawn_radius_m:
                destroy_cmds.append(carla.command.DestroyActor(v.id))
            else:
                keep.append(v)

        if destroy_cmds:
            self.client.apply_batch_sync(destroy_cmds, False)

        self.vehicles = keep
    
    def tick(self, sim_t: float):
        self._despawn_passed()

        while sim_t >= self.next_spawn_t and len(self.vehicles) < self.max_active:
            v = self._try_spawn_one()
            if v is not None:
                self.vehicles.append(v)
                self.next_spawn_t += self._sample_headway()
            else:
                # entrance blocked; retry soon without consuming a headway sample
                self.next_spawn_t = sim_t + 0.2

    def destroy_all(self):
        destroy_cmds = [carla.command.DestroyActor(v.id) for v in self.vehicles if v.is_alive]
        if destroy_cmds:
            self.client.apply_batch_sync(destroy_cmds, False)
        self.vehicles = []


@dataclass
class SceneHandles:
    ego: carla.Vehicle
    collision_sensor: carla.Sensor
    actors: List[carla.Actor]          # includes ego + traffic + sensors
    ego_start: carla.Transform
    ego_destination: carla.Location
    oncoming_anchor: carla.Transform
    traffic_stream: Optional[TrafficStream] = None

def speed_kmh(v: carla.Vehicle) -> float:
        vel = v.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def set_synchronous(world: carla.World, tm: carla.TrafficManager, fixed_delta: float = 0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_delta
    world.apply_settings(settings)

    # TM must be synchronous too if world is synchronous.
    tm.set_synchronous_mode(True)  # API requirement in sync mode. :contentReference[oaicite:3]{index=3}


def _as_maplayer(mask: int) -> carla.MapLayer:
    # Convert int bitmask -> MapLayer enum expected by the C++ API
    return carla.MapLayer(mask)

def compose_layers(*layers: carla.MapLayer) -> carla.MapLayer:
    mask = 0
    for L in layers:
        mask |= int(getattr(L, "value", L))
    return carla.MapLayer(mask)



def freeze_all_traffic_lights_green(world: carla.World):
    """
    Avoid protected left phases by forcing every traffic light to GREEN and freezing.
    CARLA exposes TrafficLight.set_state and freeze(). :contentReference[oaicite:6]{index=6}

    Note: freeze(True) is documented as freezing all traffic lights in the scene. :contentReference[oaicite:7]{index=7}
    """
    tls = world.get_actors().filter("traffic.traffic_light*")
    for tl in tls:
        tl.set_state(carla.TrafficLightState.Green)
        # These setters exist; not strictly required if we freeze.
        tl.set_green_time(10_000.0)
        tl.set_red_time(0.0)
        tl.set_yellow_time(0.0)

    # Freeze globally (calling on any TL affects all per docs)
    if len(tls) > 0:
        tls[0].freeze(True)


def draw_spawnpoint_indices(world: carla.World, life_time: float = 30.0):
    """
    Helper: label spawn points in-world so you can fly spectator and pick indices.
    This is the approach shown in CARLA's traffic manager tutorial. :contentReference[oaicite:8]{index=8}
    """
    sp = world.get_map().get_spawn_points()
    for i, t in enumerate(sp):
        world.debug.draw_string(t.location, str(i), life_time=life_time)
    print(f"Drew {len(sp)} spawn point indices for {life_time} seconds.")

def is_allowed_vehicle_bp(bp: carla.ActorBlueprint) -> bool:
    # Keep only 4-wheel road vehicles (excludes bicycles + motorcycles)
    if bp.has_attribute("number_of_wheels"):
        try:
            wheels = bp.get_attribute("number_of_wheels").as_int()
        except Exception:
            wheels = int(bp.get_attribute("number_of_wheels").value)
        if wheels != 4:
            return False
    return True

def pick_vehicle_blueprint(world: carla.World) -> carla.ActorBlueprint:
    bps = [bp for bp in world.get_blueprint_library().filter("vehicle.*") if is_allowed_vehicle_bp(bp)]
    # keep it simple/reliable
    preferred = [bp for bp in bps if "model3" in bp.id.lower()]
    bp = random.choice(preferred) if preferred else random.choice(bps)
    if bp.has_attribute("color"):
        bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
    return bp


def spawn_vehicle(world: carla.World, bp: carla.ActorBlueprint, transform: carla.Transform) -> Optional[carla.Vehicle]:
    actor = world.try_spawn_actor(bp, transform)
    return actor


def attach_collision_sensor(world: carla.World, vehicle: carla.Vehicle) -> carla.Sensor:
    bp = world.get_blueprint_library().find("sensor.other.collision")
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    return sensor


def make_oncoming_stream(
    world: carla.World,
    tm: carla.TrafficManager,
    anchor_transform: carla.Transform,
    *,
    n_vehicles: int = 25,
    spacing_m: float = 8.0,
    path_forward_m: float = 120.0,
    aggressiveness: str = "aggressive",
) -> List[carla.Vehicle]:
    """
    Spawns vehicles behind anchor_transform along the lane direction and pins them to a forward path.

    Uses TM.set_path (locations) to guide traffic, which is in the Python API. :contentReference[oaicite:9]{index=9}
    """
    m = world.get_map()
    anchor_wp = m.get_waypoint(anchor_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)

    # build a simple 2-point path: a bit ahead + far ahead
    wp_mid = anchor_wp.next(30.0)[0]
    wp_end = anchor_wp.next(path_forward_m)[0]
    path_locs = [wp_mid.transform.location, wp_end.transform.location]

    vehicles: List[carla.Vehicle] = []
    bp_lib = world.get_blueprint_library()

    # aggressiveness knobs
    # - percentage_speed_difference: negative => faster than limit (common CARLA convention)
    # - distance_to_leading_vehicle: smaller => tailgating / dense stream
    if aggressiveness == "aggressive":
        speed_diff = -30.0
        follow_dist = 1.0
    elif aggressiveness == "normal":
        speed_diff = -10.0
        follow_dist = 2.5
    else:  # cautious
        speed_diff = 10.0
        follow_dist = 4.0

    for i in range(n_vehicles):
        # place vehicles "behind" anchor
        prev_list = anchor_wp.previous((i + 1) * spacing_m)
        if not prev_list:
            continue
        wp_i = prev_list[0]
        t_i = wp_i.transform
        t_i.location.z += 0.5

        bp = pick_vehicle_blueprint(world)
        v = world.try_spawn_actor(bp, t_i)
        if v is None:
            continue

        v = v  # type: ignore
        vehicles.append(v)

        # register with TM
        v.set_autopilot(True, tm.get_port())
        tm.set_path(v, path_locs)  # :contentReference[oaicite:10]{index=10}
        tm.vehicle_percentage_speed_difference(v, speed_diff)
        tm.distance_to_leading_vehicle(v, follow_dist)

        # Optional: make oncoming really "pushy" (uncomment if you want)
        # tm.ignore_lights_percentage(v, 100.0)  # :contentReference[oaicite:11]{index=11}
        # tm.ignore_signs_percentage(v, 50.0)    # :contentReference[oaicite:12]{index=12}

    return vehicles

def reset_vehicle(world: carla.World, vehicle: carla.Vehicle, start_tf: carla.Transform):
    # Stop any controller/autopilot first
    vehicle.set_autopilot(False)
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    # Freeze physics so teleport is stable
    vehicle.set_simulate_physics(False)  # :contentReference[oaicite:8]{index=8}
    vehicle.set_transform(start_tf)      # :contentReference[oaicite:9]{index=9}
    world.tick()

    # Re-enable physics and tick once to apply cleanly
    vehicle.set_simulate_physics(True)
    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))          # m/s :contentReference[oaicite:1]{index=1}
    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))  # deg/s :contentReference[oaicite:2]{index=2}
    vehicle.disable_constant_velocity()
    world.tick()

    # Release brake so the agent can drive
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))

def build_scene(
    client: carla.Client,
    *,
    town: str,
    seed: int,
    ego_spawn_idx: int,
    ego_dest_idx: int,
    oncoming_anchor_idx: int,
    oncoming_dest_idx: int,
    n_oncoming: int,
    traffic_profile: str,
    hide_layers: bool,
    freeze_lights: bool,
    use_stream: bool = True,       # NEW
    mean_headway_s: float = 2.5,   # NEW
    burst_prob: float = 0.15,
) -> SceneHandles:
    random.seed(seed)

    # Load world (Opt map needed for layers). load_world has map_layers flags; default is All. :contentReference[oaicite:13]{index=13}
    world = client.get_world()
    current = world.get_map().name  # often like "/Game/Carla/Maps/Town01_Opt"
    if not current.endswith(town):
        if hide_layers:
            layers = compose_layers(carla.MapLayer.NONE, carla.MapLayer.Props, carla.MapLayer.Walls)
        else:
            layers = carla.MapLayer.All
        print(f"Loading map {town} (current: {current})")
        world = client.load_world(town, layers)
        
    else:
        print(f"Map already loaded: {current} (skipping load_world)")
    tm = client.get_trafficmanager(8000)
    tm.set_random_device_seed(seed)  # determinism for TM :contentReference[oaicite:14]{index=14}

    set_synchronous(world, tm, fixed_delta=0.05)
    world.tick()

    if freeze_lights:
        freeze_all_traffic_lights_green(world)
        world.tick()

    spawn_points = world.get_map().get_spawn_points()
    assert 0 <= ego_spawn_idx < len(spawn_points), "ego_spawn_idx out of range"
    assert 0 <= ego_dest_idx < len(spawn_points), "ego_dest_idx out of range"
    assert 0 <= oncoming_anchor_idx < len(spawn_points), "oncoming_anchor_idx out of range"
    assert 0 <= oncoming_dest_idx < len(spawn_points), "oncoming_dest_idx out of range"

    ego_start = spawn_points[ego_spawn_idx]
    ego_dest = spawn_points[ego_dest_idx].location
    oncoming_anchor = spawn_points[oncoming_anchor_idx]

    # Spawn ego
    ego_bp = pick_vehicle_blueprint(world)
    ego_bp.set_attribute("role_name", "hero")
    ego = world.spawn_actor(ego_bp, ego_start)  # if this fails, let it throw; you want to know
    collision_sensor = attach_collision_sensor(world, ego)

    # Spawn oncoming stream
    oncoming_dest = spawn_points[oncoming_dest_idx].location

    # Build a path that goes from anchor -> destination
    carla_map = world.get_map()
    spawn_transform = spawn_points[oncoming_anchor_idx]
    despawn_loc = spawn_points[oncoming_dest_idx].location
    anchor_wp = carla_map.get_waypoint(oncoming_anchor.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    path_locs = build_greedy_waypoint_path_locs(carla_map, oncoming_anchor.location, oncoming_dest)

    traffic_stream = None
    oncoming_cars: List[carla.Vehicle] = []

    if use_stream:
        traffic_stream = TrafficStream(
            client=client,
            world=world,
            tm=tm,
            spawn_transform=spawn_transform,
            path_locs=path_locs,
            despawn_loc=despawn_loc,
            max_active=n_oncoming,            # keep up to N active
            mean_headway_s=mean_headway_s,
            burst_prob=burst_prob,
            aggressiveness=traffic_profile,
        )
        print(f"Traffic stream enabled: max_active={n_oncoming}, mean_headway_s={mean_headway_s}, burst_prob={burst_prob}")
    else:
        # your existing one-shot spawn (kept as fallback)
        oncoming_cars = make_oncoming_stream(
            world, tm, oncoming_anchor,
            n_vehicles=n_oncoming,
            aggressiveness=traffic_profile,
        )
        print(f"Spawned oncoming vehicles: {len(oncoming_cars)} / {n_oncoming}")

    # tick once so actors fully initialize
    world.tick()

    actors = [ego, collision_sensor] + oncoming_cars
    return SceneHandles(
        ego=ego,
        collision_sensor=collision_sensor,
        actors=actors,
        ego_start=ego_start,
        ego_destination=ego_dest,
        oncoming_anchor=oncoming_anchor,
        traffic_stream=traffic_stream,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town01_Opt")
    parser.add_argument("--seed", type=int, default=0)

    # You’ll pick these using draw_spawnpoint_indices() + spectator view.
    parser.add_argument("--ego-spawn", type=int, default=12)
    parser.add_argument("--ego-dest", type=int, default=32)
    parser.add_argument("--oncoming-anchor", type=int, default=45)
    parser.add_argument("--oncoming-dest", type=int, default=227)
    parser.add_argument("--traffic-stream", action="store_true")
    parser.add_argument("--mean-headway", type=float, default=2.5)
    parser.add_argument("--burst-prob", type=float, default=0.15)

    parser.add_argument("--n-oncoming", type=int, default=25)
    parser.add_argument("--traffic-profile", choices=["cautious", "normal", "aggressive"], default="aggressive")

    parser.add_argument("--hide-layers", action="store_true")
    parser.add_argument("--freeze-lights", action="store_true")
    parser.add_argument("--show-spawn-indices", action="store_true")
    args = parser.parse_args()

    ensure_agents_importable()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0) #timeout after 1 minutes

    # Build the scene
    handles = build_scene(
        client,
        town=args.town,
        seed=args.seed,
        ego_spawn_idx=args.ego_spawn,
        ego_dest_idx=args.ego_dest,
        oncoming_anchor_idx=args.oncoming_anchor,
        oncoming_dest_idx=args.oncoming_dest,
        n_oncoming=args.n_oncoming,
        traffic_profile=args.traffic_profile,
        hide_layers=args.hide_layers,
        freeze_lights=args.freeze_lights,
        use_stream=args.traffic_stream,
        mean_headway_s=args.mean_headway,
        burst_prob=args.burst_prob,
    )

    world = client.get_world()
    if args.show_spawn_indices:
        draw_spawnpoint_indices(world, life_time=60.0)

    print("Scene ready.")
    print(f"Ego start: {handles.ego_start.location}")
    print(f"Ego dest:  {handles.ego_destination}")
    print(f"Oncoming anchor: {handles.oncoming_anchor.location}")
    print("Press Ctrl+C to quit (this script just keeps ticking so you can inspect).")

    try:
        while True:
            world.tick()
            snap = world.get_snapshot()
            sim_t = snap.timestamp.elapsed_seconds
            if handles.traffic_stream is not None:
                handles.traffic_stream.tick(sim_t)
            if handles.traffic_stream is not None and int(sim_t * 2) != int((sim_t - world.get_settings().fixed_delta_seconds) * 2):
                # ~every 0.5s of sim time
                speeds = [speed_kmh(v) for v in handles.traffic_stream.vehicles if v.is_alive]
                if speeds:
                    print(f"traffic speed km/h: mean={sum(speeds)/len(speeds):.1f} max={max(speeds):.1f} n={len(speeds)}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            world = client.get_world()
            tm = client.get_trafficmanager(8000)

            # 1) Unfreeze lights first
            tls = world.get_actors().filter("traffic.traffic_light*")
            if len(tls) > 0:
                tls[0].freeze(False)

            # 2) Return to async BEFORE destroying lots of actors
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            tm.set_synchronous_mode(False)

            # Give server one moment to apply
            world.wait_for_tick(2.0)

            # 3) Destroy stream vehicles (batch) + scene actors
            if handles.traffic_stream is not None:
                handles.traffic_stream.destroy_all()

            destroy_cmds = [carla.command.DestroyActor(a.id) for a in handles.actors if a.is_alive]
            if destroy_cmds:
                client.apply_batch_sync(destroy_cmds, False)

        except Exception as e:
            print("Cleanup exception:", e)


if __name__ == "__main__":
    main()
