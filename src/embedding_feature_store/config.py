"""Runtime settings for ``efs-migrate``.

Resolution order (highest first): explicit overrides (CLI flags), environment
variables named ``EFS_<FIELD>`` (e.g. ``EFS_BATCH_SIZE=512``), then defaults.
Environment values are strings and are parsed by the field's type
(``"true"``/``"false"`` for booleans, enum values for ``distance``).

``qdrant_url`` unset selects the in-process vector store, so a full migration
can run locally with no services.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from embedding_feature_store.schemas import (
    DistanceMetric,
    Identifier,
    MigrationPlan,
    RetryPolicy,
    VectorSpec,
)

__all__ = ["ACTIVE_COLLECTION_ALIAS", "ENV_PREFIX", "ConfigError", "Settings"]

ACTIVE_COLLECTION_ALIAS: Final = "features_active"
ENV_PREFIX: Final = "EFS_"

type Positive = Annotated[int, Field(ge=1)]


class ConfigError(ValueError):
    """Invalid configuration (maps to exit code 2)."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = None
    qdrant_timeout_s: Annotated[float, Field(gt=0)] = 10.0

    migration_id: Identifier = "efs-local"
    alias: Identifier = ACTIVE_COLLECTION_ALIAS
    blue_collection: Identifier = "features_blue"
    green_collection: Identifier = "features_green"

    source_rows: Annotated[int, Field(ge=0)] = 2000
    source_dim: Positive = 16
    blue_dim: Positive = 8
    green_dim: Positive = 32
    distance: DistanceMetric = DistanceMetric.COSINE

    batch_size: Annotated[int, Field(ge=1, le=10_000)] = 256
    queue_capacity: Positive = 4
    cdc_page_size: Positive = 512
    retry_attempts: Annotated[int, Field(ge=1, le=100)] = 5
    drain_timeout_s: Annotated[float, Field(gt=0)] = 10.0
    seed: int = 0

    live_traffic: bool = True
    hud: bool | None = None
    """None: show the HUD only when stderr is a terminal."""
    report_path: str | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @model_validator(mode="after")
    def _distinct_names(self) -> Self:
        names = {self.alias, self.blue_collection, self.green_collection}
        if len(names) != 3:  # noqa: PLR2004 - alias, blue, green
            msg = "alias, blue_collection and green_collection must be distinct"
            raise ValueError(msg)
        return self

    @classmethod
    def load(cls, env: Mapping[str, str], **overrides: object) -> Settings:
        """Build settings from ``env`` plus non-None ``overrides``; raise ConfigError."""
        values: dict[str, object] = {
            name: env[ENV_PREFIX + name.upper()]
            for name in cls.model_fields
            if ENV_PREFIX + name.upper() in env
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        try:
            return cls.model_validate(values)
        except ValueError as exc:  # pydantic.ValidationError is a ValueError
            raise ConfigError(str(exc)) from exc

    def plan(self) -> MigrationPlan:
        return MigrationPlan(
            migration_id=self.migration_id,
            source_collection=self.blue_collection,
            target_collection=self.green_collection,
            alias=self.alias,
            target_spec=VectorSpec(dimension=self.green_dim, distance=self.distance),
            batch_size=self.batch_size,
            retry=RetryPolicy(max_attempts=self.retry_attempts),
        )

    @property
    def blue_spec(self) -> VectorSpec:
        return VectorSpec(dimension=self.blue_dim, distance=self.distance)
