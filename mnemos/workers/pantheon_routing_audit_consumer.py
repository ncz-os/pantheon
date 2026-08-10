"""NATS consumer that mirrors PANTHEON routing events into the audit table."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from mnemos.core.config import get_settings
from mnemos.domain.pantheon.routing_log import (
    PANTHEON_ROUTING_SUBJECT,
    insert_routing_audit_record,
    routing_audit_record,
)

logger = logging.getLogger(__name__)
DURABLE = "mnemos_pantheon_routing_audit"


class PoisonMessageError(ValueError):
    """A routing event cannot be decoded into the published schema."""


def _decode_event(data: bytes | bytearray | memoryview | str) -> dict[str, Any]:
    try:
        raw = data if isinstance(data, str) else bytes(data).decode("utf-8")
        event = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoisonMessageError("PANTHEON routing event is not valid JSON") from exc
    if not isinstance(event, dict) or not isinstance(event.get("metadata"), dict):
        raise PoisonMessageError("PANTHEON routing event must contain metadata")
    required = ("request_id", "tenant_user_id", "alias_or_model", "resolved_to", "outcome")
    if not all(str(event.get(field) or "").strip() for field in required):
        raise PoisonMessageError("PANTHEON routing event is missing required fields")
    return event


async def _ack(msg: Any) -> None:
    callback = getattr(msg, "ack", None)
    if callback is None:
        return
    result = callback()
    if hasattr(result, "__await__"):
        await result


async def handle_message(backend: Any, msg: Any) -> None:
    event = _decode_event(getattr(msg, "data", b""))
    metadata = dict(event.pop("metadata"))
    record = routing_audit_record(event, metadata)
    if not await insert_routing_audit_record(backend, record):
        raise RuntimeError("persistence backend does not support PANTHEON routing audits")
    await _ack(msg)


async def _connect(settings: Any) -> tuple[Any, Any]:
    try:
        import nats  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("nats-py not installed") from exc
    kwargs: dict[str, Any] = {"servers": [settings.nats.url]}
    if settings.nats.token:
        kwargs["token"] = settings.nats.token
    client = await nats.connect(**kwargs)
    return client, client.jetstream()


async def _drain(client: Any, subscription: Any) -> None:
    if subscription is not None:
        try:
            await subscription.unsubscribe()
        except Exception:
            pass
    if client is not None:
        try:
            await client.drain()
        except Exception:
            try:
                await client.close()
            except Exception:
                pass


async def consumer_loop(
    backend: Any,
    *,
    settings: Any | None = None,
    retry_seconds: float = 30.0,
    connect: Callable[[Any], Awaitable[tuple[Any, Any]]] | None = None,
) -> None:
    """Consume routing events until cancelled, reconnecting after broker errors."""
    settings = settings or get_settings()
    if not settings.nats.audit_consumer_enabled:
        return
    if not settings.nats.url:
        logger.warning("PANTHEON routing audit consumer enabled without MNEMOS_NATS_URL")
        return
    connector = connect or _connect
    delay = 1.0
    while True:
        client = None
        subscription = None
        try:
            client, jetstream = await connector(settings)
            subscription = await jetstream.subscribe(
                PANTHEON_ROUTING_SUBJECT,
                durable=DURABLE,
            )
            delay = 1.0
            while True:
                msg = await subscription.next_msg(timeout=1)
                try:
                    await handle_message(backend, msg)
                except PoisonMessageError as exc:
                    logger.warning("discarding malformed PANTHEON routing event: %s", exc)
                    await _ack(msg)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("PANTHEON routing audit event write failed")
        except asyncio.CancelledError:
            await _drain(client, subscription)
            raise
        except Exception as exc:
            await _drain(client, subscription)
            logger.warning("PANTHEON routing audit consumer unavailable: %s", exc)
            await asyncio.sleep(delay)
            delay = min(retry_seconds, delay * 2)


__all__ = ["PoisonMessageError", "consumer_loop", "handle_message"]
