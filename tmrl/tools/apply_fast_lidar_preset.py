import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


FAST_LIDAR_940M_PRESET = {
    "RUN_NAME": "SAC_LIDAR_940M_FAST",
    "CUDA_TRAINING": True,
    "CUDA_INFERENCE": False,
    "CUDA_LOW_MEMORY_MODE": "AUTO",
    "CUDA_LOW_MEMORY_BATCH_SIZE": 64,
    "MAX_EPOCHS": 1000,
    "ROUNDS_PER_EPOCH": 60,
    "TRAINING_STEPS_PER_ROUND": 100,
    "MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP": 8.0,
    "ENVIRONMENT_STEPS_BEFORE_TRAINING": 200,
    "UPDATE_MODEL_INTERVAL": 100,
    "UPDATE_BUFFER_INTERVAL": 50,
    "SAVE_MODEL_EVERY": 0,
    "MEMORY_SIZE": 250000,
    "BATCH_SIZE": 64,
    "ALG": {
        "ALGORITHM": "SAC",
        "HIDDEN_SIZES": [128, 128],
        "LEARN_ENTROPY_COEF": True,
        "LR_ACTOR": 0.0003,
        "LR_CRITIC": 0.001,
        "LR_ENTROPY": 0.0003,
        "GAMMA": 0.995,
        "POLYAK": 0.995,
        "TARGET_ENTROPY": None,
        "ALPHA": 0.2,
        "ACTOR_INITIAL_GAS_BIAS": 0.8,
        "ACTOR_INITIAL_BRAKE_BIAS": -2.0,
        "ACTOR_INITIAL_STEER_BIAS": 0.0,
        "ACTOR_INITIAL_LOG_STD_BIAS": -0.7,
        "OPTIMIZER_ACTOR": "adam",
        "OPTIMIZER_CRITIC": "adam",
        "L2_ACTOR": 0.0,
        "L2_CRITIC": 0.0,
    },
    "ENV": {
        "RTGYM_INTERFACE": "TM20LIDARPROGRESS",
        "IMG_HIST_LEN": 1,
        "RTGYM_CONFIG": {
            "time_step_duration": 0.05,
            "start_obs_capture": 0.04,
            "time_step_timeout_factor": 1.0,
            "act_buf_len": 1,
            "benchmark": False,
            "wait_on_done": True,
            "ep_max_length": 1000,
            "interface_kwargs": {
                "save_replays": False,
            },
        },
        "REWARD_CONFIG": {
            "END_OF_TRACK": 100.0,
            "CONSTANT_PENALTY": 0.001,
            "CHECK_FORWARD": 500,
            "CHECK_BACKWARD": 10,
            "FAILURE_COUNTDOWN": 8,
            "MIN_STEPS": 40,
            "MAX_STRAY": 100.0,
        },
    },
}


def main():
    parser = argparse.ArgumentParser(description="Apply a fast SAC LIDAR preset for low-memory CUDA GPUs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / "TmrlData" / "config" / "config.json",
        help="Path to config.json.",
    )
    parser.add_argument(
        "--reset-training",
        action="store_true",
        help="Set RESET_TRAINING=true so an incompatible old checkpoint keeps replay memory but rebuilds the agent.",
    )
    parser.add_argument(
        "--keep-run-name",
        action="store_true",
        help="Do not replace RUN_NAME.",
    )
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, encoding="utf-8-sig") as f:
        config = json.load(f)

    preset = json.loads(json.dumps(FAST_LIDAR_940M_PRESET))
    if args.keep_run_name:
        preset.pop("RUN_NAME", None)
    preset["RESET_TRAINING"] = bool(args.reset_training)

    backup_path = config_path.with_suffix(
        config_path.suffix + "." + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bak"
    )
    shutil.copy2(config_path, backup_path)
    deep_update(config, preset)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    print(f"Applied fast LIDAR preset to {config_path}")
    print(f"Backup written to {backup_path}")


if __name__ == "__main__":
    main()
