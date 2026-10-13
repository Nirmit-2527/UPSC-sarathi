"""
upsc-sarathi-agentic :: app/services/revision.py

Spaced-repetition scheduler based on the SM-2 algorithm (SuperMemo-2).
Used to schedule revision of MCQs / facts / notes that the aspirant has
seen, so they can revise just-in-time before forgetting.

(Ported unchanged from the non-agentic reference project - spaced
repetition is a deterministic scheduling algorithm with no LLM/agent
involvement, so there was nothing to make "agentic" here.)

If progress tracking is enabled, items are persisted to DynamoDB;
otherwise an in-memory dict is used (fine for a single-user prototype).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional
import boto3
from loguru import logger

from app.config import settings


@dataclass
class Card:
    card_id: str
    topic: str
    front: str           # question
    back: str            # answer / explanation
    ef: float = 2.5      # easiness factor
    interval_days: float = 1.0
    repetitions: int = 0
    due_iso: str = ""    # next due date

    def is_due(self, now: Optional[datetime] = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return datetime.fromisoformat(self.due_iso) <= now if self.due_iso else True


# ----------------- backends -----------------
class _InMemoryBackend:
    def __init__(self) -> None:
        self._db: Dict[str, Card] = {}

    def put(self, card: Card) -> None:
        self._db[card.card_id] = card

    def get(self, card_id: str) -> Optional[Card]:
        return self._db.get(card_id)

    def list(self) -> List[Card]:
        return list(self._db.values())


class _DynamoBackend:
    def __init__(self) -> None:
        self._ddb = boto3.resource("dynamodb", region_name=settings.aws_region)
        self._table = self._ddb.Table(settings.ddb_table_progress)

    def put(self, card: Card) -> None:
        self._table.put_item(Item=asdict(card))

    def get(self, card_id: str) -> Optional[Card]:
        r = self._table.get_item(Key={"card_id": card_id})
        item = r.get("Item")
        return Card(**item) if item else None

    def list(self) -> List[Card]:
        r = self._table.scan()
        return [Card(**i) for i in r.get("Items", [])]


def _backend():
    return _DynamoBackend() if settings.enable_progress_tracking else _InMemoryBackend()


_b = None
def backend():
    global _b
    if _b is None:
        _b = _backend()
    return _b


# ----------------- SM-2 -----------------
def add_card(topic: str, front: str, back: str) -> Card:
    card = Card(
        card_id=str(uuid.uuid4()),
        topic=topic,
        front=front,
        back=back,
        due_iso=datetime.now(timezone.utc).isoformat(),
    )
    backend().put(card)
    return card


def review(card_id: str, quality: int) -> Card:
    """
    quality ∈ {0..5}  (per SM-2)
        5 - perfect recall
        4 - correct after hesitation
        3 - correct with difficulty
        2 - incorrect but recognised on review
        1 - incorrect, hard to remember
        0 - blackout
    """
    if not 0 <= quality <= 5:
        raise ValueError("quality must be in 0..5")
    c = backend().get(card_id)
    if not c:
        raise KeyError(card_id)

    if quality < 3:
        c.repetitions = 0
        c.interval_days = 1.0
    else:
        if c.repetitions == 0:
            c.interval_days = 1.0
        elif c.repetitions == 1:
            c.interval_days = 6.0
        else:
            c.interval_days = round(c.interval_days * c.ef, 2)
        c.repetitions += 1

    # update easiness factor
    c.ef = max(1.3, c.ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    c.due_iso = (datetime.now(timezone.utc) + timedelta(days=c.interval_days)).isoformat()
    backend().put(c)
    return c


def due_cards(limit: int = 20) -> List[Card]:
    return [c for c in backend().list() if c.is_due()][:limit]
