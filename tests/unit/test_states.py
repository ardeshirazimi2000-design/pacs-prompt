from __future__ import annotations

import pytest

from ctpipeline.common.errors import InvalidTransitionError
from ctpipeline.jobs.states import JobStatus as S
from ctpipeline.jobs.states import check_transition

HAPPY_PATH = [
    S.RECEIVED,
    S.VALIDATING,
    S.QUEUED,
    S.PREPROCESSING,
    S.INFERENCE,
    S.POSTPROCESSING,
    S.EXPORTING,
    S.SUCCEEDED,
]


def test_happy_path_is_allowed() -> None:
    for a, b in zip(HAPPY_PATH, HAPPY_PATH[1:], strict=False):
        check_transition(a, b)


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (S.SUCCEEDED, S.QUEUED),
        (S.FAILED, S.QUEUED),
        (S.CANCELLED, S.QUEUED),
        (S.RECEIVED, S.INFERENCE),
        (S.QUEUED, S.SUCCEEDED),
        (S.VALIDATING, S.PREPROCESSING),
    ],
)
def test_invalid_transitions(src: S, dst: S) -> None:
    with pytest.raises(InvalidTransitionError):
        check_transition(src, dst)


def test_running_stages_can_requeue_for_retry() -> None:
    for st in (S.PREPROCESSING, S.INFERENCE, S.POSTPROCESSING, S.EXPORTING):
        check_transition(st, S.QUEUED)
        check_transition(st, S.CANCELLED)


def test_empty_api_key_disables_auth() -> None:
    from ctpipeline.common.config import Settings

    assert Settings(api_key="").api_key is None
    assert Settings(api_key="k").api_key == "k"


def test_database_url_built_from_parts_escapes_password() -> None:
    from sqlalchemy.engine import make_url

    from ctpipeline.common.config import Settings

    s = Settings(db_host="postgres", db_password="p@ss:w/rd#1")
    url = make_url(s.database_url)
    assert url.host == "postgres"
    assert url.password == "p@ss:w/rd#1"
    assert url.drivername == "postgresql+psycopg"
