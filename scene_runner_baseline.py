#!/usr/bin/env python3
import os
import sys
import csv
import time
import math
import argparse
from dataclasses import asdict
from typing import List, Dict, Any, Optional

import carla

from scenarios.scene import build_scene, reset_vehicle  # type: ignore


def ensure_agents_importable():
    """Make CARLA agents/ importable (BehaviorAgent/BasicAgent) using CARLA_ROOT if needed."""
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

    # try again
    import agents  # noqa: F401


def vec_mag(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * vec_mag(v.x, v.y, v.z)


def accel_components(vehicle: carla.Vehicle) -> Dict[str, float]:
    """
    Returns longitudinal and lateral acceleration magnitudes (m/s^2), using the vehicle forward vector.
    """
    a = vehicle.get_acceleration()
    tf = vehicle.get_transform()
    f = tf.get_forward_vector()

    # normalize forward vector (usually unit, but be safe)
    fn = vec_mag(f.x, f.y, f.z)
    fx, fy, fz = (f.x / fn, f.y / fn, f.z / fn) if fn > 1e-6 else (1.0, 0.0, 0.0)

    along = a.x * fx + a.y * fy + a.z * fz
    amag2 = a.x * a.x + a.y * a.y + a.z * a.z
    lat2 = max(0.0, amag2 - along * along)
    lat = math.sqrt(lat2)
    return {"a_long": along, "a_lat": lat, "a_mag": math.sqrt(amag2)}

def warmup_traffic(world: carla.World, handles, warmup_s: float = 12.0, min_vehicles: int = 10):
    """
    Let the traffic stream build up BEFORE starting episode timing.
    Runs in sim-time, so if you tick fast it'll warm up quickly.
    """
    if handles.traffic_stream is None:
        return

    start = world.get_snapshot().timestamp.elapsed_seconds
    while True:
        snap = world.get_snapshot()
        sim_t = snap.timestamp.elapsed_seconds

        handles.traffic_stream.tick(sim_t)
        world.tick()

        n = len([v for v in handles.traffic_stream.vehicles if v.is_alive])
        if (sim_t - start) >= warmup_s and n >= min_vehicles:
            break


def run_episode(
    client: carla.Client,
    world: carla.World,
    handles,
    *,
    agent_type: str,
    behavior: str,
    timeout_s: float,
    dest_radius_m: float,
    collision_events: list,
    dt: float,
) -> Dict[str, Any]:
    """
    Runs one episode: reset ego, run baseline agent until destination reached, collision, or timeout.
    Collects comfort metrics.
    """
    collision_events.clear()
    # --- Reset episode state ---
    # Reset ego pose/velocity/physics
    reset_vehicle(world, handles.ego, handles.ego_start)

    # # Reset traffic stream (keep scene loaded; just recycle vehicles)
    # if handles.traffic_stream is not None:
    #     handles.traffic_stream.destroy_all()
    #     # reset stream timing so it starts spawning immediately
    #     # (attribute exists in your TrafficStream)
    #     handles.traffic_stream.next_spawn_t = world.get_snapshot().timestamp.elapsed_seconds

    # --- Create baseline agent ---
    ensure_agents_importable()
    if agent_type.lower() == "behavior":
        from agents.navigation.behavior_agent import BehaviorAgent  # type: ignore
        agent = BehaviorAgent(handles.ego, behavior=behavior)
    elif agent_type.lower() == "basic":
        from agents.navigation.basic_agent import BasicAgent  # type: ignore
        agent = BasicAgent(handles.ego)
    else:
        raise ValueError("agent_type must be 'behavior' or 'basic'")

    agent.set_destination(handles.ego_destination)

    # --- Metrics accumulators ---
    start_t = world.get_snapshot().timestamp.elapsed_seconds
    end_t = start_t

    max_speed = 0.0

    max_a_long = 0.0
    max_a_lat = 0.0

    max_jerk = 0.0
    jerk_sq_sum = 0.0
    jerk_n = 0
    jerk_ignore_until = world.get_snapshot().timestamp.elapsed_seconds + 0.5 #delay recording of jerk by 0.5s

    prev_a = None  # (ax, ay, az)

    success = False
    crashed = False

    # --- Run loop ---
    # Note: control applied, then tick to advance sim deterministically.
    while True:
        # spawn/despawn traffic before stepping ego (fine either way)
        snap = world.get_snapshot()
        sim_t = snap.timestamp.elapsed_seconds

        if handles.traffic_stream is not None:
            handles.traffic_stream.tick(sim_t)

        # Compute control
        # BehaviorAgent needs this each tick
        if agent_type.lower() == "behavior":
            if hasattr(agent, "update_information"):
                agent.update_information()
            elif hasattr(agent, "_update_information"):
                agent._update_information()  # fallback for some versions

        control = agent.run_step()
        handles.ego.apply_control(control)

        # Advance sim one step
        world.tick()
        snap = world.get_snapshot()
        sim_t = snap.timestamp.elapsed_seconds
        end_t = sim_t

        # Update metrics after advancing
        spd = speed_kmh(handles.ego)
        max_speed = max(max_speed, spd)

        ac = accel_components(handles.ego)
        max_a_long = max(max_a_long, abs(ac["a_long"]))
        max_a_lat = max(max_a_lat, ac["a_lat"])

        a = handles.ego.get_acceleration()
        if prev_a is not None and sim_t >= jerk_ignore_until:
            jx = (a.x - prev_a[0]) / dt
            jy = (a.y - prev_a[1]) / dt
            jz = (a.z - prev_a[2]) / dt
            j = vec_mag(jx, jy, jz)
            max_jerk = max(max_jerk, j)
            jerk_sq_sum += j * j
            jerk_n += 1
        prev_a = (a.x, a.y, a.z)

        # Termination checks
        if len(collision_events) > 0:
            crashed = True
            break

        if handles.ego.get_location().distance(handles.ego_destination) <= dest_radius_m:
            success = True
            break

        if (sim_t - start_t) >= timeout_s:
            break

    rms_jerk = math.sqrt(jerk_sq_sum / jerk_n) if jerk_n > 0 else 0.0

    return {
        "success": success,
        "crashed": crashed,
        "time_s": (end_t - start_t),
        "max_speed_kmh": max_speed,
        "max_a_long_mps2": max_a_long,
        "max_a_lat_mps2": max_a_lat,
        "max_jerk_mps3": max_jerk,
        "rms_jerk_mps3": rms_jerk,
        "n_collisions": len(collision_events),
    }


def write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)

    # Scene args (same ones you’ve been using)
    parser.add_argument("--town", default="Town01_Opt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hide-layers", action="store_true")
    parser.add_argument("--freeze-lights", action="store_true")

    parser.add_argument("--ego-spawn", type=int, default=224)
    parser.add_argument("--ego-dest", type=int, default=83)
    parser.add_argument("--oncoming-anchor", type=int, default=173)
    parser.add_argument("--oncoming-dest", type=int, default=227)
    parser.add_argument("--n-oncoming", type=int, default=30)
    parser.add_argument("--traffic-stream", action="store_true")
    parser.add_argument("--mean-headway", type=float, default=2.2)
    parser.add_argument("--burst-prob", type=float, default=0.2)
    parser.add_argument("--traffic-profile", choices=["cautious", "normal", "aggressive"], default="normal")

    # Runner / baseline
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agent", choices=["behavior", "basic"], default="behavior")
    parser.add_argument("--behavior", choices=["cautious", "normal", "aggressive"], default="normal")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--dest-radius-m", type=float, default=5.0)

    # Output
    parser.add_argument("--out", default="results/baseline_results.csv")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    # Build scene ONCE (map load, layers, lights, traffic stream, spawn ego)
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
    collision_events = []
    # Attach listener ONCE. Clear collision_events at episode start.
    handles.collision_sensor.listen(lambda e: collision_events.append(e))
    settings = world.get_settings()
    dt = settings.fixed_delta_seconds if settings.fixed_delta_seconds else 0.05  # fallback

    results: List[Dict[str, Any]] = []

    try:
        # Prime one tick so everything is initialized
        world.tick()

        warmup_traffic(world, handles, warmup_s=5.0, min_vehicles=3)

        for ep in range(args.episodes):
            r = run_episode(
                client, world, handles,
                agent_type=args.agent,
                behavior=args.behavior,
                timeout_s=args.timeout_s,
                collision_events = collision_events,
                dest_radius_m=args.dest_radius_m,
                dt=dt,
            )
            r.update({
                "episode": ep,
                "agent": args.agent,
                "behavior": args.behavior,
                "traffic_profile": args.traffic_profile,
                "mean_headway": args.mean_headway,
                "burst_prob": args.burst_prob,
                "n_oncoming": args.n_oncoming,
                "ego_spawn": args.ego_spawn,
                "ego_dest": args.ego_dest,
                "oncoming_anchor": args.oncoming_anchor,
                "oncoming_dest": args.oncoming_dest,
            })
            results.append(r)
            print(f"[ep {ep}] success={r['success']} crashed={r['crashed']} time_s={r['time_s']:.2f} max_jerk={r['max_jerk_mps3']:.2f}")

        write_csv(args.out, results)
        print(f"Wrote: {args.out}")

    finally:
        # Cleanup: restore async + destroy actors to keep CARLA stable for the next run
        try:
            tm = client.get_trafficmanager(8000)
            tls = world.get_actors().filter("traffic.traffic_light*")
            if len(tls) > 0:
                tls[0].freeze(False)

            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
            tm.set_synchronous_mode(False)

            world.wait_for_tick(2.0)

            if getattr(handles, "traffic_stream", None) is not None:
                handles.traffic_stream.destroy_all()

            destroy_cmds = [carla.command.DestroyActor(a.id) for a in handles.actors if a.is_alive]
            if destroy_cmds:
                client.apply_batch_sync(destroy_cmds, False)

        except Exception as e:
            print("Cleanup exception:", e)


if __name__ == "__main__":
    main()
