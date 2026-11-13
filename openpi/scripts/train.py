import dataclasses
import datetime as _datetime
import functools
import json
import logging
import os
import platform
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

try:
    import psutil as _psutil
except Exception:  # pragma: no cover - optional runtime dependency.
    _psutil = None


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _now_iso() -> str:
    return _datetime.datetime.now().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _selected_attrs(obj: Any, names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name in names:
        if hasattr(obj, name):
            result[name] = _jsonable(getattr(obj, name))
    return result


def _run_command(args: list[str], *, timeout: float = 5.0) -> list[str]:
    try:
        output = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if output.returncode != 0:
        return []
    return [line.strip() for line in output.stdout.splitlines() if line.strip()]


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _memory_snapshot() -> dict[str, Any]:
    if _psutil is not None:
        mem = _psutil.virtual_memory()
        return {
            "total_gb": mem.total / 1024**3,
            "available_gb": mem.available / 1024**3,
            "used_gb": mem.used / 1024**3,
            "percent": mem.percent,
        }

    meminfo_path = pathlib.Path("/proc/meminfo")
    if not meminfo_path.exists():
        return {}

    values: dict[str, float] = {}
    for line in meminfo_path.read_text().splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = float(raw_value.strip().split()[0]) * 1024

    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return {}
    used = total - available
    return {
        "total_gb": total / 1024**3,
        "available_gb": available / 1024**3,
        "used_gb": used / 1024**3,
        "percent": used / total * 100.0,
    }


def _existing_path(path: pathlib.Path) -> pathlib.Path:
    path = path.resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _disk_snapshot(path: pathlib.Path) -> dict[str, Any]:
    try:
        usage = shutil.disk_usage(_existing_path(path))
    except OSError:
        return {}
    return {
        "path": str(path),
        "total_gb": usage.total / 1024**3,
        "used_gb": usage.used / 1024**3,
        "free_gb": usage.free / 1024**3,
        "percent": usage.used / usage.total * 100.0 if usage.total else None,
    }


def _gpu_static_snapshot() -> list[dict[str, Any]]:
    lines = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mb": _float_or_none(parts[2]),
                "power_limit_w": _float_or_none(parts[3]),
            }
        )
    return gpus


def _gpu_runtime_snapshot() -> list[dict[str, Any]]:
    lines = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        gpus.append(
            {
                "index": parts[0],
                "utilization_gpu_percent": _float_or_none(parts[1]),
                "memory_used_mb": _float_or_none(parts[2]),
                "memory_total_mb": _float_or_none(parts[3]),
                "power_draw_w": _float_or_none(parts[4]),
                "temperature_c": _float_or_none(parts[5]),
            }
        )
    return gpus


def _config_snapshot(config: _config.TrainConfig, *, resuming: bool) -> dict[str, Any]:
    model = config.model
    data = config.data
    device_count = jax.device_count()
    data_parallel_groups = device_count // config.fsdp_devices if config.fsdp_devices else None
    return {
        "config_name": config.name,
        "exp_name": config.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "assets_dir": str(config.assets_dirs),
        "batch_size": config.batch_size,
        "num_train_steps": config.num_train_steps,
        "save_interval": config.save_interval,
        "log_interval": config.log_interval,
        "num_workers": config.num_workers,
        "fsdp_devices": config.fsdp_devices,
        "jax_device_count": device_count,
        "jax_local_device_count": jax.local_device_count(),
        "jax_process_count": jax.process_count(),
        "data_parallel_groups": data_parallel_groups,
        "batch_per_data_parallel_group": config.batch_size // data_parallel_groups if data_parallel_groups else None,
        "resume": config.resume,
        "overwrite": config.overwrite,
        "resuming_from_checkpoint": resuming,
        "model": _selected_attrs(
            model,
            (
                "model_type",
                "action_dim",
                "action_horizon",
                "max_token_len",
                "dtype",
                "paligemma_variant",
                "action_expert_variant",
            ),
        ),
        "data": _selected_attrs(
            data,
            (
                "repo_id",
                "data_root",
                "use_tactile",
                "force_predict",
                "force_guide",
                "camera_keys",
            ),
        ),
    }


def _static_system_snapshot(config: _config.TrainConfig) -> dict[str, Any]:
    data_root = getattr(config.data, "data_root", None)
    data_disk = _disk_snapshot(pathlib.Path(data_root)) if isinstance(data_root, (str, os.PathLike)) else {}
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env": {
            key: os.environ.get(key)
            for key in (
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "XLA_PYTHON_CLIENT_MEM_FRACTION",
                "TF_ENABLE_ONEDNN_OPTS",
                "XLA_FLAGS",
                "JAX_PLATFORMS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "memory": _memory_snapshot(),
        "checkpoint_disk": _disk_snapshot(config.checkpoint_dir),
        "data_disk": data_disk,
        "gpus": _gpu_static_snapshot(),
        "argv": sys.argv,
    }


def _runtime_resource_snapshot() -> dict[str, Any]:
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else None
    disk_io = None
    cpu_percent = None
    if _psutil is not None:
        cpu_percent = _psutil.cpu_percent(interval=None)
        counters = _psutil.disk_io_counters()
        if counters is not None:
            disk_io = {"read_bytes": counters.read_bytes, "write_bytes": counters.write_bytes}

    snapshot = {
        "timestamp": _now_iso(),
        "cpu_percent": cpu_percent,
        "load_avg": list(load_avg) if load_avg is not None else None,
        "memory": _memory_snapshot(),
        "gpus": _gpu_runtime_snapshot(),
        "disk_io": disk_io,
    }
    return snapshot


def _add_disk_io_rates(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    elapsed_s: float | None,
) -> None:
    if not previous or not elapsed_s or elapsed_s <= 0:
        return
    current_io = current.get("disk_io")
    previous_io = previous.get("disk_io")
    if not current_io or not previous_io:
        return
    current["disk_io_rate"] = {
        "read_mb_s": (current_io["read_bytes"] - previous_io["read_bytes"]) / elapsed_s / 1024**2,
        "write_mb_s": (current_io["write_bytes"] - previous_io["write_bytes"]) / elapsed_s / 1024**2,
    }


def _timing_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg_s": 0.0, "p95_s": 0.0, "max_s": 0.0}
    return {
        "avg_s": float(np.mean(values)),
        "p95_s": float(np.percentile(values, 95)),
        "max_s": float(np.max(values)),
    }


def _resource_summary(resources: dict[str, Any]) -> str:
    parts = []
    if resources.get("cpu_percent") is not None:
        parts.append(f"cpu={resources['cpu_percent']:.1f}%")
    elif resources.get("load_avg"):
        cpu_count = os.cpu_count() or 1
        parts.append(f"load1={resources['load_avg'][0]:.1f}/{cpu_count}")

    memory = resources.get("memory") or {}
    if memory.get("used_gb") is not None and memory.get("total_gb") is not None:
        parts.append(f"ram={memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB")

    gpus = resources.get("gpus") or []
    gpu_utils = [gpu["utilization_gpu_percent"] for gpu in gpus if gpu.get("utilization_gpu_percent") is not None]
    gpu_used = [gpu["memory_used_mb"] for gpu in gpus if gpu.get("memory_used_mb") is not None]
    gpu_total = [gpu["memory_total_mb"] for gpu in gpus if gpu.get("memory_total_mb") is not None]
    if gpu_utils:
        parts.append(f"gpu_util_avg={float(np.mean(gpu_utils)):.1f}%")
    if gpu_used and gpu_total:
        parts.append(f"gpu_mem={sum(gpu_used) / 1024:.1f}/{sum(gpu_total) / 1024:.1f}GB")

    disk_io_rate = resources.get("disk_io_rate") or {}
    if disk_io_rate:
        parts.append(f"disk_read={disk_io_rate['read_mb_s']:.1f}MB/s")
        parts.append(f"disk_write={disk_io_rate['write_mb_s']:.1f}MB/s")

    return ", ".join(parts) if parts else "n/a"


def _write_profile_event(checkpoint_dir: pathlib.Path, event: dict[str, Any]) -> None:
    event = {"timestamp": _now_iso(), **event}
    with open(checkpoint_dir / "training_profile.jsonl", "a") as f:
        f.write(json.dumps(_jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")


def _append_metrics_header(config: _config.TrainConfig, *, resuming: bool) -> None:
    event = {
        "event": "run_start",
        "config": _config_snapshot(config, resuming=resuming),
        "system": _static_system_snapshot(config),
    }
    with open(config.checkpoint_dir / "training_metrics.log", "a") as f:
        f.write("\n=== RUN_PROFILE_BEGIN ===\n")
        f.write(json.dumps(_jsonable(event), ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n=== RUN_PROFILE_END ===\n")
    _write_profile_event(config.checkpoint_dir, event)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    if jax.process_index() == 0:
        _append_metrics_header(config, resuming=resuming)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    run_wall_start = time.time()
    last_resource_snapshot = _runtime_resource_snapshot()
    last_resource_time = time.time()
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        desc=f"{config.name}/{config.exp_name}",
        unit="step",
        dynamic_ncols=True,
    )

    infos = []
    data_times = []
    step_times = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            t_model_start = time.time()
            train_state, info = ptrain_step(train_rng, train_state, batch)
            info = jax.tree.map(lambda x: x.block_until_ready(), info)
            step_times.append(time.time() - t_model_start)
        
        infos.append(info)
        
        if step % config.log_interval == 0:
            completed_steps = int(jax.device_get(train_state.step))
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            
            data_stats = _timing_stats(data_times)
            model_stats = _timing_stats(step_times)
            total_avg = data_stats["avg_s"] + model_stats["avg_s"]
            steps_per_hour = 3600.0 / total_avg if total_avg > 0 else 0.0
            samples_per_second = config.batch_size / total_avg if total_avg > 0 else 0.0

            resources = _runtime_resource_snapshot()
            resource_time = time.time()
            _add_disk_io_rates(resources, last_resource_snapshot, resource_time - last_resource_time)
            last_resource_snapshot = resources
            last_resource_time = resource_time

            time_str = (
                f"[Avg over {len(step_times)} steps] "
                f"Data avg/p95/max: {data_stats['avg_s']:.3f}/{data_stats['p95_s']:.3f}/{data_stats['max_s']:.3f}s | "
                f"Model avg/p95/max: {model_stats['avg_s']:.3f}/{model_stats['p95_s']:.3f}/{model_stats['max_s']:.3f}s | "
                f"Total avg: {total_avg:.3f}s | "
                f"steps/h: {steps_per_hour:.1f} | samples/s: {samples_per_second:.1f}"
            )
            metric_values = {k: float(np.asarray(v)) for k, v in reduced_info.items()}
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in metric_values.items())
            final_log = (
                f"Config {config.name} | Exp {config.exp_name} | "
                f"Completed {completed_steps:06d}/{config.num_train_steps:06d} steps | "
                f"Loop step {step:06d}: {time_str} || {info_str} || resources: {_resource_summary(resources)}"
            )
            
            pbar.write(final_log)
            
            if jax.process_index() == 0:
                log_file = config.checkpoint_dir / "training_metrics.log"
                with open(log_file, "a") as f:
                    f.write(final_log + "\n")
                _write_profile_event(
                    config.checkpoint_dir,
                    {
                        "event": "step",
                        "step": int(step),
                        "completed_steps": completed_steps,
                        "total_steps": config.num_train_steps,
                        "config_name": config.name,
                        "exp_name": config.exp_name,
                        "timing": {
                            "num_steps": len(step_times),
                            "data": data_stats,
                            "model": model_stats,
                            "total_avg_s": total_avg,
                            "steps_per_hour": steps_per_hour,
                            "samples_per_second": samples_per_second,
                        },
                        "metrics": metric_values,
                        "resources": resources,
                    },
                )
            
            wandb.log(
                {
                    **reduced_info,
                    "timing/data_avg_s": data_stats["avg_s"],
                    "timing/model_avg_s": model_stats["avg_s"],
                    "timing/total_avg_s": total_avg,
                    "speed/steps_per_hour": steps_per_hour,
                    "speed/samples_per_second": samples_per_second,
                },
                step=step,
            )
            infos = []
            data_times = []
            step_times = []
            
        t_data_start = time.time()
        batch = next(data_iter)
        data_times.append(time.time() - t_data_start)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if jax.process_index() == 0:
        _write_profile_event(
            config.checkpoint_dir,
            {
                "event": "run_end",
                "final_step": int(train_state.step),
                "elapsed_wall_s": time.time() - run_wall_start,
                "resources": _runtime_resource_snapshot(),
            },
        )


if __name__ == "__main__":
    main(_config.cli())
