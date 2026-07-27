import requests

BASE_URL = "http://localhost:8000"
MODEL = "llama3.2:3b"
FINE_TUNED_MODEL = "customer-v1"

def test_chat_success():
    payload = {
        "query_id": "q-001",
        "messages": [{"role": "user", "content": "Hello, what are your business hours?"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload, timeout=120)
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_id"] == "q-001"
    assert "reply" in body
    assert body["model"]
    assert isinstance(body["input_tokens"], int)
    assert isinstance(body["output_tokens"], int)


def test_chat_with_system_prompt():
    payload = {
        "query_id": "q-002",
        "messages": [{"role": "user", "content": "Hi"}],
        "system_prompt": "You are a helpful customer service agent.",
    }
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload, timeout=120)
    assert resp.status_code == 200
    assert resp.json()["query_id"] == "q-002"


def test_chat_with_context():
    payload = {
        "query_id": "q-004",
        "messages": [
            {"role": "user", "content": "What is your return policy?"},
            {"role": "assistant", "content": "You can return items within 30 days."},
            {"role": "user", "content": "Can I get a refund instead?"},
        ],
    }
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload, timeout=120)
    assert resp.status_code == 200
    assert resp.json()["query_id"] == "q-004"


def test_chat_missing_messages():
    payload = {"query_id": "q-005"}
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload)
    assert resp.status_code == 422


def test_chat_missing_query_id():
    payload = {"messages": [{"role": "user", "content": "Hello"}]}
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload)
    assert resp.status_code == 422


def test_chat_invalid_role():
    payload = {
        "query_id": "q-006",
        "messages": [{"role": "moderator", "content": "Hello"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json=payload)
    assert resp.status_code == 422


def test_chat_empty_body():
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}", json={})
    assert resp.status_code == 422


def test_chat_stream_success():
    payload = {
        "query_id": "q-007",
        "messages": [{"role": "user", "content": "Tell me a short joke"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}/stream", json=payload, stream=True, timeout=120)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
    chunks = list(resp.iter_content(decode_unicode=True))
    assert len(chunks) > 0


def test_chat_stream_invalid_body():
    resp = requests.post(f"{BASE_URL}/chat/{MODEL}/stream", json={})
    assert resp.status_code == 422


def test_chat_fine_tuned_model():
    payload = {
        "query_id": "q-008",
        "messages": [{"role": "user", "content": "How do I reset my password?"}],
    }
    resp = requests.post(f"{BASE_URL}/chat/{FINE_TUNED_MODEL}", json=payload, timeout=120)
    assert resp.status_code == 200
    body = resp.json()
    assert body["query_id"] == "q-008"
    assert "reply" in body
