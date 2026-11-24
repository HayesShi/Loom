"""Compare layout and materialization policies over a stream of versions."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PolicyCost:
    """Costs in a common time unit for one build, each publish and each read.

    Build includes one-off preparation. Publish is charged once per analyzed
    version and read is charged for each request to that version.
    """

    build: float
    publish: float
    read: float

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (self.build, self.publish, self.read)):
            raise ValueError("costs must be finite and nonnegative")

    def total(self, versions: int, reads_per_version: int) -> float:
        if versions < 1 or reads_per_version < 1:
            raise ValueError("versions and reads_per_version must be positive")
        return self.build + versions * (self.publish + reads_per_version * self.read)

    def crossover(self, baseline: "PolicyCost", *, versions: int) -> int | None:
        """Smallest positive read count per version with strictly lower cost."""
        if self.total(versions, 1) < baseline.total(versions, 1):
            return 1
        saving = baseline.read - self.read
        if saving <= 0:
            return None
        overhead = (self.build - baseline.build) / versions + self.publish - baseline.publish
        return max(2, math.floor(overhead / saving) + 1)
