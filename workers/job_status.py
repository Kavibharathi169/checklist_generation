import json
import os
import time
from typing import Any

import redis


_client = None


def _get_redis_client() -> redis.Redis:
    global _client
    if _client is None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        _client = redis.Redis.from_url(redis_url, decode_responses=True)
    return _client


def _status_key(job_id: str) -> str:
    return f"job_status:{job_id}"


def set_status(job_id: str, payload: dict[str, Any]) -> None:
    ttl = int(os.getenv("JOB_STATUS_TTL_SECONDS", "86400"))
    payload = {
        **payload,
        "updated_at": int(time.time()),
    }
    client = _get_redis_client()
    client.setex(_status_key(job_id), ttl, json.dumps(payload))


def get_status(job_id: str) -> dict[str, Any] | None:
    client = _get_redis_client()
    raw = client.get(_status_key(job_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None
