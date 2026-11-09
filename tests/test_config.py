"""Settings: defaults, EFS_* environment parsing, override precedence, validation."""

from __future__ import annotations

import pytest

from embedding_feature_store.config import ACTIVE_COLLECTION_ALIAS, ConfigError, Settings
from embedding_feature_store.schemas import DistanceMetric, VectorSpec


def test_defaults_and_plan() -> None:
    settings = Settings.load({})
    assert settings.alias == ACTIVE_COLLECTION_ALIAS == "features_active"
    assert settings.qdrant_url is None
    assert settings.hud is None
    plan = settings.plan()
    assert (plan.source_collection, plan.target_collection, plan.alias) == (
        "features_blue",
        "features_green",
        "features_active",
    )
    assert plan.target_spec == VectorSpec(dimension=32, distance=DistanceMetric.COSINE)
    assert plan.retry.max_attempts == settings.retry_attempts == 5
    assert settings.blue_spec == VectorSpec(dimension=8, distance=DistanceMetric.COSINE)


def test_environment_strings_are_parsed() -> None:
    settings = Settings.load(
        {
            "EFS_BATCH_SIZE": "512",
            "EFS_LIVE_TRAFFIC": "false",
            "EFS_HUD": "true",
            "EFS_DISTANCE": "dot",
            "EFS_DRAIN_TIMEOUT_S": "2.5",
            "EFS_QDRANT_URL": "http://qdrant:6333",
            "EFS_QDRANT_API_KEY": "s3cret",
            "UNRELATED": "ignored",
        }
    )
    assert settings.batch_size == 512
    assert settings.live_traffic is False
    assert settings.hud is True
    assert settings.distance is DistanceMetric.DOT
    assert settings.drain_timeout_s == 2.5
    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_api_key is not None
    assert settings.qdrant_api_key.get_secret_value() == "s3cret"
    assert "s3cret" not in repr(settings), "secrets never appear in reprs or logs"


def test_overrides_beat_environment_and_none_means_unset() -> None:
    settings = Settings.load({"EFS_BATCH_SIZE": "512", "EFS_SEED": "9"}, batch_size=64, seed=None)
    assert (settings.batch_size, settings.seed) == (64, 9)


@pytest.mark.parametrize(
    ("env", "fragment"),
    [
        ({"EFS_BATCH_SIZE": "0"}, "batch_size"),
        ({"EFS_BATCH_SIZE": "lots"}, "batch_size"),
        ({"EFS_DISTANCE": "manhattan"}, "distance"),
        ({"EFS_GREEN_COLLECTION": "features_blue"}, "distinct"),
        ({"EFS_ALIAS": "has space"}, "alias"),
        ({"EFS_LOG_LEVEL": "TRACE"}, "log_level"),
        ({"EFS_DRAIN_TIMEOUT_S": "0"}, "drain_timeout_s"),
    ],
)
def test_invalid_configuration(env: dict[str, str], fragment: str) -> None:
    with pytest.raises(ConfigError, match=fragment):
        Settings.load(env)


def test_unknown_override_rejected() -> None:
    with pytest.raises(ConfigError, match="bogus"):
        Settings.load({}, bogus=1)


def test_config_error_is_a_value_error() -> None:
    assert issubclass(ConfigError, ValueError)
