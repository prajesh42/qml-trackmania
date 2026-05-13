import logging

import torch


def _cuda_memory_mb(device):
    try:
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        total_mb = props.total_memory // (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(index) // (1024 * 1024)
        allocated_mb = torch.cuda.memory_allocated(index) // (1024 * 1024)
        free_estimate_mb = max(0, total_mb - reserved_mb)
        return total_mb, free_estimate_mb, allocated_mb
    except Exception as exc:
        logging.warning("Could not inspect CUDA memory for %s (%s).", device, exc)
        return None, None, None


def cuda_total_memory_mb(device="cuda"):
    total_mb, _, _ = _cuda_memory_mb(device)
    return total_mb


def is_low_memory_cuda(device, threshold_mb=4096):
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    total_mb = cuda_total_memory_mb(dev)
    return total_mb is not None and total_mb <= int(threshold_mb)


def resolve_torch_device(requested_device=None, role="torch", min_free_memory_mb=0):
    requested = "auto" if requested_device is None else str(requested_device).strip().lower()
    if requested in {"", "none"}:
        requested = "auto"

    wants_cuda = requested == "auto" or requested.startswith("cuda")
    if wants_cuda and torch.cuda.is_available():
        device = "cuda" if requested == "auto" else requested
        total_mb, free_mb, allocated_mb = _cuda_memory_mb(device)
        if min_free_memory_mb and free_mb is not None and free_mb < int(min_free_memory_mb):
            logging.warning(
                "%s requested %s, but estimated free CUDA memory is %s MiB "
                "(minimum configured: %s MiB). Falling back to CPU.",
                role,
                device,
                free_mb,
                min_free_memory_mb,
            )
            return "cpu"
        if total_mb is not None:
            logging.info(
                "%s using %s with %s MiB total, ~%s MiB free, %s MiB allocated.",
                role,
                device,
                total_mb,
                free_mb,
                allocated_mb,
            )
        return device

    if requested.startswith("cuda"):
        logging.warning("%s requested %s, but CUDA is unavailable. Falling back to CPU.", role, requested)
    return "cpu"


def effective_batch_size(configured_batch_size, device, low_memory_mode="AUTO", low_memory_batch_size=None):
    configured_batch_size = int(configured_batch_size)
    if configured_batch_size <= 0:
        raise ValueError("BATCH_SIZE must be positive")

    mode = str(low_memory_mode).strip().upper()
    low_memory = mode in {"1", "TRUE", "YES", "ON"}
    if mode == "AUTO":
        low_memory = is_low_memory_cuda(device)

    if not low_memory:
        return configured_batch_size

    cap = int(low_memory_batch_size) if low_memory_batch_size is not None else 64
    if cap <= 0:
        raise ValueError("CUDA_LOW_MEMORY_BATCH_SIZE must be positive")

    resolved = min(configured_batch_size, cap)
    if resolved != configured_batch_size:
        logging.warning(
            "Low-memory CUDA mode capped training batch size from %s to %s.",
            configured_batch_size,
            resolved,
        )
    return resolved
