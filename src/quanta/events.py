from __future__ import annotations

from typing import Any

from quanta.db import Store


class EventSink:
    def __init__(self, store: Store):
        self.store = store

    def emit(self, session_id: str, event_type: str, **payload: Any) -> int:
        return self.store.append_event(session_id, event_type, payload)
