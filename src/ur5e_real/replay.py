from __future__ import annotations

import csv
import math
import socket
import time
from dataclasses import dataclass
from pathlib import Path

from .control.chunk import ChunkStreamConfig, stream_tcp_chunk
from .control.gripper_policy import GripperCommandConfig, GripperPolicy
from .control.servoj import ServoJController, ServoJStreamConfig, start_servoj_program
from .data.session_manifest import load_manifest
from .hardware.gripper import GripperSerial
from .hardware.urscript import move_linear, pose_literal

POSE_COLUMNS = ["tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz"]


@dataclass(frozen=True)
class ReplayConfig:
    action_csv: Path
    gripper_events: Path | None
    robot_host: str
    robot_port: int = 30001
    socket_timeout_s: float = 10.0
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 9600
    serial_timeout_s: float = 1.0
    row_stride: int = 5
    max_segments: int | None = None
    execute: bool = False
    backend: str = "socket"
    enable_gripper: bool = True
    go_to_start: bool = True
    acceleration: float = 0.4
    velocity_min: float = 0.02
    velocity_max: float = 0.12
    blend_radius: float = 0.008
    minimum_translation_m: float = 0.003
    minimum_rotation_rad: float = 0.03
    assumed_point_dt_s: float = 0.5
    wait_scale: float = 1.35
    wait_extra_s: float = 0.25
    rtde_port: int = 30004
    policy_hz: float = 10.0
    chunk_size: int = 6
    chunk_gap_s: float = 0.08
    max_chunks: int = 0
    servoj_frequency_hz: int = 500
    servoj_config_xml: Path | None = None
    servoj_program_script: Path | None = None
    servoj_mode: int = 2
    servoj_connect_timeout_s: float = 15.0
    servoj_lookahead_time: float = 0.1
    servoj_gain: int = 300
    rtde_max_linear_velocity: float = 0.40


def resolve_replay_paths(source: Path, gripper_events: Path | None = None) -> tuple[Path, Path | None]:
    """Resolve either a session manifest or the legacy pair of CSV paths."""
    if source.suffix.lower() != ".json":
        return source, gripper_events
    manifest = load_manifest(source)
    paths = manifest.get("paths") or {}
    action_value = paths.get("rtde")
    if not action_value:
        raise ValueError(f"{source}: paths.rtde is missing")

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (source.parent / path).resolve()

    action_path = resolve(str(action_value))
    if gripper_events is not None:
        return action_path, gripper_events
    events_value = paths.get("gripper_events")
    return action_path, resolve(str(events_value)) if events_value else None


def load_action_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = ["controller_time_s", *POSE_COLUMNS]
        missing = [name for name in required if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        for index, row in enumerate(reader):
            rows.append(
                {
                    "row_index": index,
                    "controller_time_s": float(row["controller_time_s"]),
                    "pose": [float(row[name]) for name in POSE_COLUMNS],
                    "gripper_state": (
                        float(row["gripper_state"])
                        if row.get("gripper_state") not in (None, "")
                        else None
                    ),
                }
            )
    return rows


def load_gripper_events(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    events: list[dict] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            event = row.get("event", "").strip().lower()
            if event in {"open", "close"}:
                events.append({"controller_time_s": float(row["controller_time_s"]), "event": event})
    return sorted(events, key=lambda item: item["controller_time_s"])


def _distance(a: list[float], b: list[float], start: int, end: int) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(start, end)))


def translation_distance(a: list[float], b: list[float]) -> float:
    return _distance(a, b, 0, 3)


def rotation_distance(a: list[float], b: list[float]) -> float:
    return _distance(a, b, 3, 6)


def smooth_rotation_vectors(rows: list[dict]) -> int:
    corrected = 0
    if len(rows) < 2:
        return corrected
    previous = rows[0]["pose"][3:6]
    for row in rows[1:]:
        rotation = row["pose"][3:6]
        norm = math.sqrt(sum(value**2 for value in rotation))
        candidates = [rotation]
        if norm > 1e-9:
            unit = [value / norm for value in rotation]
            candidates.extend(
                [
                    [rotation[index] - 2.0 * math.pi * unit[index] for index in range(3)],
                    [rotation[index] + 2.0 * math.pi * unit[index] for index in range(3)],
                ]
            )
        best = min(candidates, key=lambda item: _distance(item, previous, 0, 3))
        if _distance(best, rotation, 0, 3) > 1e-6:
            row["pose"][3:6] = best
            corrected += 1
        previous = row["pose"][3:6]
    return corrected


def _event_neighbor_indices(rows: list[dict], events: list[dict]) -> set[int]:
    keep = {0, len(rows) - 1} if rows else set()
    for event in events:
        timestamp = event["controller_time_s"]
        before = None
        after = None
        for index, row in enumerate(rows):
            if row["controller_time_s"] <= timestamp:
                before = index
            if after is None and row["controller_time_s"] >= timestamp:
                after = index
                break
        if before is not None:
            keep.add(before)
        if after is not None:
            keep.add(after)
    return keep


def filter_waypoints(rows: list[dict], events: list[dict], min_translation: float, min_rotation: float) -> list[dict]:
    if len(rows) < 2:
        return list(rows)
    forced = _event_neighbor_indices(rows, events)
    result: list[dict] = []
    last: dict | None = None
    for index, row in enumerate(rows):
        keep = index in forced or last is None
        if last is not None:
            keep = keep or translation_distance(last["pose"], row["pose"]) >= min_translation
            keep = keep or rotation_distance(last["pose"], row["pose"]) >= min_rotation
        if keep:
            result.append(row)
            last = row
    return result


def build_segments(rows: list[dict], events: list[dict], first_target_index: int = 1) -> list[dict]:
    segments: list[dict] = []
    start = first_target_index
    for event in events:
        end = -1
        for index, row in enumerate(rows):
            if row["controller_time_s"] <= event["controller_time_s"]:
                end = index
            else:
                break
        segments.append({"start": start, "end": end, "event_after": event})
        start = max(start, end + 1)
    if start < len(rows):
        segments.append({"start": start, "end": len(rows) - 1, "event_after": None})
    return segments


def build_segment_program(rows: list[dict], segment: dict, index: int, cfg: ReplayConfig) -> tuple[str, float]:
    if segment["end"] < segment["start"]:
        return "", 0.0
    lines = [f"def ur5e_replay_segment_{index:03d}():"]
    estimated_s = 0.0
    for target_index in range(segment["start"], segment["end"] + 1):
        distance = translation_distance(rows[target_index - 1]["pose"], rows[target_index]["pose"])
        velocity = max(cfg.velocity_min, min(cfg.velocity_max, distance / cfg.assumed_point_dt_s))
        blend = 0.0
        if target_index < segment["end"]:
            next_distance = translation_distance(rows[target_index]["pose"], rows[target_index + 1]["pose"])
            if distance >= 0.002 and next_distance >= 0.002:
                blend = min(cfg.blend_radius, 0.3 * distance, 0.3 * next_distance)
        lines.append(
            f"  movel({pose_literal(rows[target_index]['pose'])}, "
            f"a={cfg.acceleration:.4f}, v={velocity:.4f}, r={blend:.5f})"
        )
        estimated_s += distance / max(velocity, 1e-6)
    lines.append("end")
    return "\n".join(lines) + "\n", max(0.5, estimated_s * cfg.wait_scale + cfg.wait_extra_s)


def build_rtde_chunks(
    rows: list[dict],
    events: list[dict],
    *,
    chunk_size: int = 6,
    first_target_index: int = 1,
    max_chunks: int = 0,
) -> list[list[dict]]:
    """Pair recorded events with 10 Hz targets, then group them like DP actions."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if max_chunks < 0:
        raise ValueError("max_chunks must be non-negative")
    targets: list[dict] = []
    event_index = 0
    for row in rows[first_target_index:]:
        row_events: list[dict] = []
        while event_index < len(events) and events[event_index]["controller_time_s"] <= row["controller_time_s"]:
            row_events.append(events[event_index])
            event_index += 1
        targets.append({"row": row, "events": row_events})
    chunks = [targets[index : index + chunk_size] for index in range(0, len(targets), chunk_size)]
    return chunks[:max_chunks] if max_chunks else chunks


def _run_socket_replay(cfg: ReplayConfig) -> None:
    if cfg.row_stride < 1:
        raise ValueError("row_stride must be at least one")
    rows = load_action_rows(cfg.action_csv)[:: cfg.row_stride]
    if not rows:
        raise RuntimeError("action CSV contains no selected rows")
    events = [
        event
        for event in load_gripper_events(cfg.gripper_events)
        if rows[0]["controller_time_s"] <= event["controller_time_s"] <= rows[-1]["controller_time_s"]
    ]
    corrected = smooth_rotation_vectors(rows)
    filtered = filter_waypoints(rows, events, cfg.minimum_translation_m, cfg.minimum_rotation_rad)
    segments = build_segments(filtered, events, 1 if cfg.go_to_start else 0)
    if cfg.max_segments is not None:
        segments = segments[: cfg.max_segments]

    print(f"rows: {len(rows)} -> {len(filtered)}; rotation corrections: {corrected}")
    print(f"events: {len(events)}; segments: {len(segments)}; execute: {cfg.execute}")
    programs = [build_segment_program(filtered, segment, index, cfg) for index, segment in enumerate(segments)]
    for index, (segment, (_, wait_s)) in enumerate(zip(segments, programs)):
        targets = max(0, segment["end"] - segment["start"] + 1)
        event = segment["event_after"]
        print(f"segment {index}: targets={targets}, wait={wait_s:.2f}s, event={None if event is None else event['event']}")
    if not cfg.execute:
        print("[DRY RUN] add --execute only after reviewing this summary")
        return

    robot = socket.create_connection((cfg.robot_host, cfg.robot_port), timeout=cfg.socket_timeout_s)
    gripper = (
        GripperSerial(cfg.serial_port, cfg.serial_baudrate, cfg.serial_timeout_s)
        if cfg.enable_gripper
        else None
    )
    try:
        if cfg.go_to_start:
            command = f"movel({pose_literal(filtered[0]['pose'])}, a=0.4, v=0.04)\n"
            robot.sendall(command.encode("utf-8"))
            time.sleep(3.0)
        for segment, (program, wait_s) in zip(segments, programs):
            if program:
                robot.sendall(program.encode("utf-8"))
                time.sleep(wait_s)
            event = segment["event_after"]
            if event is not None and gripper is not None:
                gripper.close() if event["event"] == "close" else gripper.open()
                time.sleep(0.8)
    finally:
        if gripper is not None:
            gripper.shutdown()
        robot.close()


def _run_rtde_replay(cfg: ReplayConfig) -> None:
    if cfg.policy_hz <= 0:
        raise ValueError("policy_hz must be positive")
    if cfg.chunk_gap_s < 0:
        raise ValueError("chunk_gap_s must be non-negative")
    if cfg.rtde_max_linear_velocity <= 0:
        raise ValueError("rtde_max_linear_velocity must be positive")

    rows = load_action_rows(cfg.action_csv)
    if not rows:
        raise RuntimeError("action CSV contains no rows")
    events = [
        event
        for event in load_gripper_events(cfg.gripper_events)
        if rows[0]["controller_time_s"] <= event["controller_time_s"] <= rows[-1]["controller_time_s"]
    ]
    corrected = smooth_rotation_vectors(rows)
    chunks = build_rtde_chunks(
        rows,
        events,
        chunk_size=cfg.chunk_size,
        first_target_index=1 if cfg.go_to_start else 0,
        max_chunks=cfg.max_chunks,
    )
    target_count = sum(len(chunk) for chunk in chunks)
    mapped_event_count = sum(len(target["events"]) for chunk in chunks for target in chunk)
    if not chunks:
        raise RuntimeError("action CSV contains no RTDE replay targets")
    estimated_s = target_count / cfg.policy_hz + max(0, len(chunks) - 1) * cfg.chunk_gap_s
    print(
        f"rows: {len(rows)}; targets: {target_count}; rotation corrections: {corrected}; "
        f"events: {mapped_event_count}"
    )
    print(
        f"backend: rtde; chunks: {len(chunks)} (size={cfg.chunk_size}); "
        f"action_dt={1.0 / cfg.policy_hz:.3f}s; chunk_gap={cfg.chunk_gap_s:.3f}s; "
        f"trajectory_time={estimated_s:.2f}s; execute: {cfg.execute}"
    )
    if not cfg.execute:
        print("[DRY RUN] add --execute only after reviewing this summary")
        return
    if cfg.servoj_config_xml is None or cfg.servoj_program_script is None:
        raise RuntimeError("RTDE replay requires the servoJ XML and robot program paths")

    if cfg.go_to_start:
        print("[ALIGN] low-speed socket moveL to the first recorded pose")
        move_linear(
            cfg.robot_host,
            rows[0]["pose"],
            acceleration=0.4,
            velocity=0.04,
            port=cfg.robot_port,
            timeout_s=cfg.socket_timeout_s,
        )
        time.sleep(3.0)

    controller = ServoJController(
        ServoJStreamConfig(
            robot_host=cfg.robot_host,
            robot_port=cfg.rtde_port,
            config_xml_path=cfg.servoj_config_xml,
            frequency_hz=cfg.servoj_frequency_hz,
            servoj_mode=cfg.servoj_mode,
            connect_timeout_s=cfg.servoj_connect_timeout_s,
        )
    )
    stream_config = ChunkStreamConfig(
        policy_hz=cfg.policy_hz,
        servo_hz=cfg.servoj_frequency_hz,
        max_linear_velocity=cfg.rtde_max_linear_velocity,
    )
    gripper = (
        GripperSerial(cfg.serial_port, cfg.serial_baudrate, cfg.serial_timeout_s)
        if cfg.enable_gripper
        else None
    )
    gripper_policy = GripperPolicy(gripper, GripperCommandConfig()) if gripper is not None else None
    try:
        controller.connect_and_prime()
        start_servoj_program(
            robot_host=cfg.robot_host,
            robot_port=cfg.robot_port,
            socket_timeout_s=cfg.socket_timeout_s,
            program_script=cfg.servoj_program_script,
            lookahead_time=cfg.servoj_lookahead_time,
            gain=cfg.servoj_gain,
        )
        controller.activate()
        print(
            f"[EXECUTE] recorded actions through the DP RTDE servoJ path; "
            f"linear limit={cfg.rtde_max_linear_velocity:g}m/s"
        )
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index and cfg.chunk_gap_s:
                time.sleep(cfg.chunk_gap_s)
            print(f"[CHUNK {chunk_index}] {len(chunk)} recorded actions")

            def apply_events(action_index: int, _target: list[float]) -> None:
                if gripper_policy is None:
                    return
                action = chunk[action_index]
                gripper_state = action["row"]["gripper_state"]
                if gripper_state is not None:
                    gripper_policy.step(float(gripper_state))
                    return
                for event in action["events"]:
                    gripper.close() if event["event"] == "close" else gripper.open()

            stream_tcp_chunk(
                controller,
                [target["row"]["pose"] for target in chunk],
                stream_config,
                on_waypoint=apply_events,
            )
        print(f"[DONE] {target_count} recorded actions in {len(chunks)} chunks")
    except KeyboardInterrupt:
        print("\n[STOP] RTDE replay interrupted")
    finally:
        controller.stop()
        if gripper is not None:
            gripper.shutdown()


def run_replay(cfg: ReplayConfig) -> None:
    if cfg.backend == "socket":
        _run_socket_replay(cfg)
    elif cfg.backend == "rtde":
        _run_rtde_replay(cfg)
    else:
        raise ValueError(f"unsupported replay backend: {cfg.backend}")
