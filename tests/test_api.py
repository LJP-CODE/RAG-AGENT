"""Integration tests for a running RAG-Agent API service."""

import os

import pytest
import requests


BASE_URL = os.getenv("RAG_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.getenv("RAG_AGENT_API_KEY")


@pytest.fixture(scope="module")
def api_headers():
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        response = requests.get(f"{BASE_URL}/health", headers=headers, timeout=3)
    except requests.RequestException as exc:
        pytest.skip(f"RAG-Agent API is not running at {BASE_URL}: {exc}")

    if response.status_code != 200:
        pytest.skip(
            f"RAG-Agent API health check returned {response.status_code}; "
            "check RAG_AGENT_API_KEY if API key authentication is enabled"
        )
    return headers


@pytest.mark.parametrize(
    "question",
    [
        "显卡PCB一般多少层？",
        "什么是底部填充胶？",
        "显存和数据线之间为什么要等长绕线？",
    ],
)
def test_ask_returns_an_answer(api_headers, question):
    response = requests.post(
        f"{BASE_URL}/ask",
        headers=api_headers,
        json={"question": question, "temperature": 0.1},
        timeout=120,
    )

    assert response.status_code == 200, response.text
    assert response.json().get("answer", "").strip()
