from enum import IntEnum
from typing import Any

## Tiers

TIERS = frozenset({"green", "yellow", "red"})
TIER_UNKNOWN = "unknown"


class Tier(IntEnum):
    green = 0
    yellow = 1
    red = 2


## Common

DomainResult = dict[str, Any]
