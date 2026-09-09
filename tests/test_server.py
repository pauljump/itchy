"""test_server.py — Integration tests for Usual WebMCP, OpenAPI, and Web Studio server."""

import json
import threading
import time
import urllib.request
import urllib.error
import pytest

from usual.server import ThreadingHTTPServer, UsualHandler, ConsumerUsualHandler, GLOBAL_STORE


@pytest.fixture(scope="module")
def live_server():
    GLOBAL_STORE.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), UsualHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_web_studio_page(live_server):
    req = urllib.request.Request(f"{live_server}/studio")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "Usual Studio" in html
        assert "The Dilemma Ritual" in html
        assert "Drop History" in html


def test_consumer_onboarding_http(live_server):
    with urllib.request.urlopen(f"{live_server}/") as resp:
        assert resp.status == 200
        assert "Help ChatGPT get to know you" in resp.read().decode()
    with urllib.request.urlopen(f"{live_server}/api/onboarding/questions") as resp:
        questions = json.load(resp)
        assert len(questions) == 3
    before = GLOBAL_STORE.get_entries()
    request = urllib.request.Request(
        f"{live_server}/api/onboarding/prepare",
        data=json.dumps({"answers": [{"id": "writing", "call": "Warm and simple"}]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as resp:
        assert resp.headers["Cache-Control"] == "no-store"
        result = json.load(resp)
        assert result["entries"][0]["call"] == "Warm and simple"
    assert GLOBAL_STORE.get_entries() == before


def test_public_mode_only_exposes_stateless_consumer_routes():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ConsumerUsualHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            html = response.read().decode()
            assert 'href="/studio"' not in html
            assert "server does not save your answers" in html
            assert "analytics do not include your preference answers" in html
            assert response.headers["Cache-Control"] == "no-store"
        with urllib.request.urlopen(base + "/api/onboarding/questions") as response:
            assert len(json.load(response)) == 3
        request = urllib.request.Request(base + "/api/onboarding/prepare",
            data=json.dumps({"answers": [{"id": "writing", "call": "Warm and simple"}]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["entries"][0]["call"] == "Warm and simple"
            assert response.headers.get("Access-Control-Allow-Origin") is None
        for method, path in [
            ("GET", "/studio"), ("GET", "/api/corpus"), ("GET", "/api/export"),
            ("GET", "/sse"), ("GET", "/openapi.json"), ("HEAD", "/api/corpus"),
            ("POST", "/message"), ("POST", "/api/interview/answer"),
            ("POST", "/api/mine/text"), ("POST", "/api/exam/build"),
            ("DELETE", "/api/corpus/item?id=test"),
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(base + path, method=method))
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_openapi_spec(live_server):
    req = urllib.request.Request(f"{live_server}/openapi.json")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        spec = json.loads(resp.read().decode("utf-8"))
        assert spec["openapi"] == "3.1.0"
        assert "/api/interview/answer" in spec["paths"]
        assert "/api/export" in spec["paths"]


def test_interview_and_corpus_api(live_server):
    # 1. Get questions
    req = urllib.request.Request(f"{live_server}/api/interview/questions")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        questions = json.loads(resp.read().decode("utf-8"))
        assert len(questions) > 0

    # 2. Submit answer
    ans_payload = json.dumps({
        "question_id": "P1",
        "call": "Ship in 2 days rough",
        "reasons": "Speed to feedback is everything",
        "quote": "I will always ship in 2 days rough."
    }).encode("utf-8")
    post_req = urllib.request.Request(
        f"{live_server}/api/interview/answer",
        data=ans_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(post_req) as resp:
        assert resp.status == 200
        res = json.loads(resp.read().decode("utf-8"))
        assert res["ok"] is True
        assert res["total_corpus"] >= 1

    # 3. Check corpus
    with urllib.request.urlopen(f"{live_server}/api/corpus") as resp:
        corpus = json.loads(resp.read().decode("utf-8"))
        assert len(corpus) >= 1
        assert corpus[0]["call"] == "Ship in 2 days rough"

    # 4. Check export
    with urllib.request.urlopen(f"{live_server}/api/export?format=chatgpt") as resp:
        text = resp.read().decode("utf-8")
        assert "# How to Decide & Respond" in text
        assert "Ship in 2 days rough" in text


def test_webmcp_sse_handshake_and_tool_call(live_server):
    # 1. Initiate SSE connection
    req = urllib.request.Request(f"{live_server}/sse")
    resp = urllib.request.urlopen(req)

    # Read endpoint event
    line1 = resp.readline().decode("utf-8").strip()
    line2 = resp.readline().decode("utf-8").strip()
    resp.readline()  # empty separator

    assert line1 == "event: endpoint"
    endpoint_path = line2.replace("data: ", "")
    assert "/message?sessionId=" in endpoint_path

    # Extract sessionId
    session_id = endpoint_path.split("sessionId=")[1]

    # 2. Send JSON-RPC initialize
    init_payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    }).encode("utf-8")

    post_req = urllib.request.Request(
        f"{live_server}/message?sessionId={session_id}",
        data=init_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(post_req) as post_resp:
        assert post_resp.status == 202

    # Read SSE message event
    ev_line = resp.readline().decode("utf-8").strip()
    data_line = resp.readline().decode("utf-8").strip()
    resp.readline()  # separator

    assert ev_line == "event: message"
    rpc_res = json.loads(data_line.replace("data: ", ""))
    assert rpc_res["id"] == 1
    assert rpc_res["result"]["serverInfo"]["name"] == "usual-mcp"

    # 3. Send tools/list
    tools_payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }).encode("utf-8")

    post_req2 = urllib.request.Request(
        f"{live_server}/message?sessionId={session_id}",
        data=tools_payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(post_req2) as post_resp2:
        assert post_resp2.status == 202

    # Read SSE message event
    ev_line2 = resp.readline().decode("utf-8").strip()
    data_line2 = resp.readline().decode("utf-8").strip()
    resp.readline()

    assert ev_line2 == "event: message"
    rpc_res2 = json.loads(data_line2.replace("data: ", ""))
    assert rpc_res2["id"] == 2
    tool_names = [t["name"] for t in rpc_res2["result"]["tools"]]
    assert "usual_get_interview_question" in tool_names
    assert "usual_record_judgment" in tool_names
    assert "usual_export_instructions" in tool_names

    resp.close()
