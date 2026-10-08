"""Regression tests for configuration and guardrail behavior."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config_loader import DataConfig
from app.agent_guardrails import AgentGuardrails


def test_docker_data_paths_are_local_outside_container():
    config = DataConfig({
        "chroma_db_dir": "/app/data/chroma_db",
        "audit_log_dir": "/app/data/audit_logs",
    })

    assert config.chroma_db_dir.endswith("data/chroma_db")
    assert config.audit_log_dir.endswith("data/audit_logs")
    assert not config.chroma_db_dir.startswith("/app/")


def test_guardrails_reject_dangerous_input():
    guardrails = AgentGuardrails({"sensitive_words": ["测试敏感词"]})

    safe, _, rules = guardrails.filter_input("DROP TABLE users")
    assert not safe
    assert "sql_drop_table" in rules

    safe, _, rules = guardrails.filter_input("包含测试敏感词")
    assert not safe
    assert any(rule.startswith("sensitive_word:") for rule in rules)
