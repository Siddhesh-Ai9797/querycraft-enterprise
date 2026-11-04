"""Safety rules of the SQL validator. These run without a database."""
from unittest.mock import MagicMock

import pytest

from agents.sql_validator import SQLValidatorAgent


@pytest.fixture
def validator():
    # The EXPLAIN layer is mocked to always pass, so only the static rules are tested here.
    return SQLValidatorAgent(MagicMock())


@pytest.mark.parametrize("sql", [
    "DROP TABLE artist",
    "DELETE FROM artist",
    "UPDATE artist SET name = 'x'",
    "INSERT INTO artist VALUES (1, 'x')",
    "TRUNCATE artist",
])
def test_rejects_non_select(validator, sql):
    assert validator.validate(sql)["valid"] is False


@pytest.mark.parametrize("sql", [
    "SELECT * FROM artist; DROP TABLE artist",
    "SELECT * FROM artist; DELETE FROM album",
    "select name from artist where 1=1; update artist set name = 'x'",
])
def test_rejects_hidden_write_after_select(validator, sql):
    assert validator.validate(sql)["valid"] is False


def test_rejects_empty(validator):
    assert validator.validate("   ")["valid"] is False


def test_allows_plain_select(validator):
    assert validator.validate("SELECT name FROM artist LIMIT 5")["valid"] is True


def test_keyword_inside_a_name_is_not_blocked(validator):
    # "updated_at" contains UPDATE but is a column name, not a statement.
    assert validator.validate("SELECT updated_at FROM invoice")["valid"] is True
