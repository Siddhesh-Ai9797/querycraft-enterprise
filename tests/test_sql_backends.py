"""Unit tests for the shared SQL cleanup and the remote model adapter. No network."""
from types import SimpleNamespace

import pytest

from core.sql_text import build_prompt, clean_sql
from core import remote_sql_adapter
from core.remote_sql_adapter import RemoteSQLAdapter


def test_double_quoted_values_become_single_quoted():
    assert clean_sql('SELECT * FROM genre WHERE name = "Rock"') == "SELECT * FROM genre WHERE name = 'Rock'"


def test_no_backslashes_left_behind():
    assert "\\" not in clean_sql('SELECT 1 WHERE a = "x" AND b = "y"')


def test_keeps_first_statement_and_strips_fences():
    raw = "```sql\nSELECT COUNT(*)\nFROM artist;\nSELECT 2;\n```"
    assert clean_sql(raw) == "SELECT COUNT(*) FROM artist;"


def test_empty_reply():
    assert clean_sql("") == "" and clean_sql("   \n ") == ""


def test_prompt_matches_training_template():
    prompt = build_prompt("How many artists?", "CREATE TABLE artist (id INT)")
    assert prompt.startswith("### System:\n")
    assert "### Schema:\nCREATE TABLE artist (id INT)\n\n" in prompt
    assert prompt.endswith("### Question:\nHow many artists?\n\n### SQL:\n")


class _FakeClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sent = []
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="querycraft-sql")]))
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(text=' SELECT name FROM genre WHERE name = "Rock"')],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=12),
        )


@pytest.fixture
def fake_openai(monkeypatch):
    clients = []

    def factory(**kwargs):
        client = _FakeClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(remote_sql_adapter, "OpenAI", factory)
    return clients


def test_remote_adapter_sends_raw_prompt_to_completions(fake_openai, monkeypatch):
    monkeypatch.delenv("VLLM_URL", raising=False)
    adapter = RemoteSQLAdapter("vllm")
    adapter.load()
    out = adapter.generate("Rock tracks?", "CREATE TABLE genre (name TEXT)")
    sent = fake_openai[0].sent[0]
    assert sent["prompt"].endswith("### SQL:\n")
    assert sent["temperature"] == 0
    assert "\n\n" in sent["stop"]
    assert out["sql"] == "SELECT name FROM genre WHERE name = 'Rock'"
    assert out["prompt_tokens"] == 120
    assert fake_openai[0].kwargs["base_url"] == "http://localhost:8001/v1"


def test_modal_reads_its_own_url_and_key(fake_openai, monkeypatch):
    monkeypatch.setenv("MODAL_URL", "https://x--querycraft-sql-serve.modal.run/v1")
    monkeypatch.setenv("MODAL_API_KEY", "secret")
    RemoteSQLAdapter("modal")
    assert fake_openai[0].kwargs["base_url"].startswith("https://x--")
    assert fake_openai[0].kwargs["api_key"] == "secret"
    assert fake_openai[0].kwargs["timeout"] == 180.0


def test_modal_without_url_fails_clearly(fake_openai, monkeypatch):
    monkeypatch.delenv("MODAL_URL", raising=False)
    monkeypatch.delenv("SQL_MODEL_URL", raising=False)
    with pytest.raises(ValueError, match="MODAL_URL"):
        RemoteSQLAdapter("modal")


def test_wrong_model_name_is_caught_on_load(fake_openai):
    adapter = RemoteSQLAdapter("llamacpp", model="something-else")
    with pytest.raises(RuntimeError):
        adapter.load()


def test_benchmark_scores_the_whole_statement():
    from benchmarks.sql_benchmark import full_statement, normalize
    raw = "```sql\nSELECT MIN(game)\nFROM table_name_61\nWHERE opponent = 'phoenix';\n```"
    assert full_statement(raw) == "SELECT MIN(game) FROM table_name_61 WHERE opponent = 'phoenix'"
    assert full_statement(" SELECT a FROM t\n\nextra text") == "SELECT a FROM t"
    assert normalize("SELECT a  FROM t;") == normalize("select a from t")
