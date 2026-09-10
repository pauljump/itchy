"""server.py — Zero-dependency WebMCP, OpenAPI, and Web Studio server.

Runs on Python 3.11+ standard library. Provides:
  1. WebMCP (Model Context Protocol over SSE + JSON-RPC 2.0 at /sse and /message)
  2. OpenAPI 3.1 schema and REST endpoints for ChatGPT Custom GPT Actions (/api/... and /openapi.json)
  3. Built-in Usual Web Studio (single-page interactive UI at /)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import queue
import re
import sys
import threading
import urllib.parse
import uuid
from importlib.resources import files
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any

from .interview import list_questions, get_question, answer_to_judgment_entry
from .parsers import parse_conversation_export, extract_candidate_judgments
from .redact import redact_entries
from .onboarding import QUESTIONS as ONBOARDING_QUESTIONS, prepare_answers
from .synthesize import (
    synthesize_chatgpt_instructions,
    synthesize_markdown_knowledge,
    synthesize_system_prompt,
)


# ---------------------------------------------------------------------------
# In-Memory Session & Corpus Store
# ---------------------------------------------------------------------------

class UsualStore:
    """Thread-safe store holding the active judgment corpus and exam state."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.entries: list[dict[str, Any]] = []
        self.exam_items: list[dict[str, Any]] = []
        self.exam_answers: list[dict[str, Any]] = []
        self.sse_sessions: dict[str, queue.Queue[dict[str, Any]]] = {}

    def get_entries(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.entries)

    def add_entries(self, new_entries: list[dict[str, Any]]) -> int:
        with self.lock:
            redacted = redact_entries(new_entries)
            count = 0
            for item in redacted:
                # Ensure id
                if "id" not in item:
                    item["id"] = f"w1b1:{len(self.entries) + 1}"
                # Deduplicate by situation + call
                exists = any(
                    e.get("situation") == item.get("situation") and
                    e.get("call") == item.get("call")
                    for e in self.entries
                )
                if not exists:
                    self.entries.append(item)
                    count += 1
            return count

    def delete_entry(self, entry_id: str) -> bool:
        with self.lock:
            initial = len(self.entries)
            self.entries = [e for e in self.entries if e.get("id") != entry_id]
            return len(self.entries) < initial

    def clear(self) -> None:
        with self.lock:
            self.entries.clear()
            self.exam_items.clear()
            self.exam_answers.clear()

    def build_exam(self, holdout_count: int = 5) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        with self.lock:
            if len(self.entries) < 3:
                # Fall back to returning whatever we have
                holdout = list(self.entries)
            else:
                k = min(holdout_count, max(1, len(self.entries) // 4))
                # Deterministic pick: every 4th item
                holdout = [e for i, e in enumerate(self.entries) if i % 4 == 0][:k]

            self.exam_answers = holdout
            # Strip call and reasons from items
            self.exam_items = [
                {
                    "id": e.get("id", f"item_{i}"),
                    "situation": e.get("situation", ""),
                    "domain": e.get("domain", "other"),
                    "date": e.get("date", ""),
                }
                for i, e in enumerate(holdout)
            ]
            return list(self.exam_items), list(self.exam_answers)

    def grade_predictions(self, predictions: list[dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            answers_map = {a["id"]: a for a in self.exam_answers}
            scored: list[dict[str, Any]] = []

            for p in predictions:
                pid = p.get("id")
                ans = answers_map.get(pid)
                if not ans:
                    continue

                pred_call = str(p.get("call") or p.get("predicted_call") or "").lower()
                real_call = str(ans.get("call", "")).lower()

                # Call match heuristic
                if not pred_call:
                    c_match = 0
                elif real_call in pred_call or pred_call in real_call:
                    c_match = 2
                elif any(word in pred_call for word in real_call.split() if len(word) > 4):
                    c_match = 1
                else:
                    c_match = 0

                # Reason match heuristic
                real_reasons = [str(r).lower() for r in ans.get("reasons", [])]
                pred_reasons = p.get("reasons") or p.get("predicted_reasons") or []
                if isinstance(pred_reasons, str):
                    pred_reasons = [pred_reasons]
                pred_reasons_str = " ".join(str(r).lower() for r in pred_reasons)

                if not real_reasons:
                    r_match = None
                else:
                    matched = sum(
                        1 for r in real_reasons
                        if any(w in pred_reasons_str for w in r.split() if len(w) > 4)
                    )
                    if matched == len(real_reasons):
                        r_match = 2
                    elif matched > 0:
                        r_match = 1
                    else:
                        r_match = 0

                scored.append({
                    "id": pid,
                    "call_match": c_match,
                    "reason_match": r_match,
                    "note": f"Compared prediction against ground-truth call: '{ans.get('call')}'"
                })

            if not scored:
                return {"call_pct": 0.0, "reason_pct": 0.0, "items": []}

            call_sum = sum(s["call_match"] for s in scored)
            call_pct = round(call_sum / (len(scored) * 2) * 100, 1)

            reason_items = [s["reason_match"] for s in scored if s["reason_match"] is not None]
            if reason_items:
                reason_pct = round(sum(reason_items) / (len(reason_items) * 2) * 100, 1)
            else:
                reason_pct = None

            return {
                "call_pct": call_pct,
                "reason_pct": reason_pct,
                "scored_items": scored
            }

    # SSE session management
    def create_sse_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sse_sessions[session_id] = queue.Queue()
        return session_id

    def get_sse_queue(self, session_id: str) -> queue.Queue[dict[str, Any]] | None:
        with self.lock:
            return self.sse_sessions.get(session_id)

    def remove_sse_session(self, session_id: str) -> None:
        with self.lock:
            self.sse_sessions.pop(session_id, None)


GLOBAL_STORE = UsualStore()


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------

MCP_TOOLS = [
    {
        "name": "usual_get_interview_question",
        "description": "Fetch a calibrated dilemma question to interview the user about their decision style and values.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain filter (product, design, factory, money, people, voice)."},
                "id": {"type": "string", "description": "Optional question ID (e.g. 'P1')."}
            }
        }
    },
    {
        "name": "usual_record_judgment",
        "description": "Record a verified judgment entry (situation, call, reasons, domain) into the user's judgment corpus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "situation": {"type": "string", "description": "Neutral description of the decision context."},
                "call": {"type": "string", "description": "The specific decision or choice the user made."},
                "reasons": {"type": "array", "items": {"type": "string"}, "description": "The reasons or principles stated by the user."},
                "domain": {"type": "string", "enum": ["product", "design", "factory", "money", "people", "voice", "other"]},
                "quote": {"type": "string", "description": "Verbatim user quote anchoring this decision."}
            },
            "required": ["situation", "call", "domain"]
        }
    },
    {
        "name": "usual_mine_text",
        "description": "Extract decision atoms from a conversation transcript, pasted text, or notes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The conversation text to parse."},
                "source_label": {"type": "string", "description": "Source identifier, default 'chat_paste'."}
            },
            "required": ["text"]
        }
    },
    {
        "name": "usual_get_corpus",
        "description": "Retrieve all verified judgment entries currently in the corpus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Optional domain filter."}
            }
        }
    },
    {
        "name": "usual_build_exam",
        "description": "Split the corpus and generate a blinded prediction exam with held-out situations for testing model alignment.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "holdout_size": {"type": "integer", "description": "Number of items to hold out (default 5)."}
            }
        }
    },
    {
        "name": "usual_grade_prediction",
        "description": "Score a student prediction against held-out ground truth using the harsh-on-generic rubric.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "call": {"type": "string"},
                            "reasons": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["id", "call"]
                    }
                }
            },
            "required": ["predictions"]
        }
    },
    {
        "name": "usual_export_instructions",
        "description": "Generate compiled instructions (ChatGPT Custom Instructions, Markdown knowledge, or XML prompt).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["chatgpt", "markdown", "xml"],
                    "description": "Target format (default 'chatgpt')."
                }
            }
        }
    }
]


def handle_mcp_tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute an MCP tool call and return response content."""
    if name == "usual_get_interview_question":
        qid = arguments.get("id")
        if qid:
            q = get_question(qid)
        else:
            domain = arguments.get("domain")
            questions = list_questions(domain)
            q = questions[0] if questions else None

        if not q:
            return {"content": [{"type": "text", "text": "No questions found matching criteria."}]}
        return {"content": [{"type": "text", "text": json.dumps(q, indent=2)}]}

    elif name == "usual_record_judgment":
        reasons = arguments.get("reasons", [])
        if isinstance(reasons, str):
            reasons = [reasons]
        entry = {
            "situation": arguments.get("situation", ""),
            "call": arguments.get("call", ""),
            "reasons": reasons,
            "domain": arguments.get("domain", "other"),
            "date": datetime.date.today().isoformat(),
            "provenance": {
                "file": "mcp_tool",
                "quote": arguments.get("quote", arguments.get("call", ""))
            },
            "confidence": 1.0
        }
        added = GLOBAL_STORE.add_entries([entry])
        return {
            "content": [{
                "type": "text",
                "text": f"Recorded judgment entry (id: {entry.get('id')}). Active corpus size: {len(GLOBAL_STORE.get_entries())}."
            }]
        }

    elif name == "usual_mine_text":
        text = arguments.get("text", "")
        # Synthesize a single conversation wrapper
        conv = {
            "title": "Pasted Notes",
            "date": datetime.date.today().isoformat(),
            "turns": [{"role": "user", "text": text}]
        }
        candidates = extract_candidate_judgments([conv], source_label=arguments.get("source_label", "chat_paste"))
        added = GLOBAL_STORE.add_entries(candidates)
        return {
            "content": [{
                "type": "text",
                "text": f"Mined {len(candidates)} candidate(s), added {added} clean non-duplicate judgment(s). Total corpus: {len(GLOBAL_STORE.get_entries())}."
            }]
        }

    elif name == "usual_get_corpus":
        domain = arguments.get("domain")
        entries = GLOBAL_STORE.get_entries()
        if domain:
            entries = [e for e in entries if e.get("domain") == domain]
        return {"content": [{"type": "text", "text": json.dumps(entries, indent=2)}]}

    elif name == "usual_build_exam":
        size = arguments.get("holdout_size", 5)
        items, answers = GLOBAL_STORE.build_exam(holdout_count=size)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "exam_items": items,
                    "instructions": "Predict the user's call and reasons for each situation. Then submit to usual_grade_prediction."
                }, indent=2)
            }]
        }

    elif name == "usual_grade_prediction":
        preds = arguments.get("predictions", [])
        result = GLOBAL_STORE.grade_predictions(preds)
        return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}

    elif name == "usual_export_instructions":
        fmt = arguments.get("format", "chatgpt")
        entries = GLOBAL_STORE.get_entries()
        if fmt == "chatgpt":
            txt = synthesize_chatgpt_instructions(entries)
        elif fmt == "markdown":
            txt = synthesize_markdown_knowledge(entries)
        else:
            txt = synthesize_system_prompt(entries)
        return {"content": [{"type": "text", "text": txt}]}

    else:
        return {"content": [{"type": "text", "text": f"Unknown tool: {name}"}], "isError": True}


# ---------------------------------------------------------------------------
# OpenAPI 3.1 Specification
# ---------------------------------------------------------------------------

OPENAPI_SPEC = {
    "openapi": "3.1.0",
    "info": {
        "title": "Usual API",
        "description": "Universal Judgment Corpus API for forging personal alignment, running blinded fidelity exams, and exporting ChatGPT Custom Instructions.",
        "version": "1.1.0"
    },
    "servers": [
        {"url": "http://localhost:8780", "description": "Local Usual instance"}
    ],
    "paths": {
        "/api/interview/questions": {
            "get": {
                "operationId": "getInterviewQuestions",
                "summary": "Get dilemma interview questions",
                "responses": {
                    "200": {
                        "description": "List of questions",
                        "content": {"application/json": {"schema": {"type": "array"}}}
                    }
                }
            }
        },
        "/api/interview/answer": {
            "post": {
                "operationId": "submitInterviewAnswer",
                "summary": "Submit an answer to a dilemma question",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "question_id": {"type": "string"},
                                    "call": {"type": "string"},
                                    "reasons": {"type": "string"},
                                    "quote": {"type": "string"}
                                },
                                "required": ["question_id", "call"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Answer saved",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            }
        },
        "/api/corpus": {
            "get": {
                "operationId": "getCorpus",
                "summary": "Get all verified judgment entries",
                "responses": {
                    "200": {
                        "description": "Active corpus",
                        "content": {"application/json": {"schema": {"type": "array"}}}
                    }
                }
            }
        },
        "/api/exam/build": {
            "post": {
                "operationId": "buildExam",
                "summary": "Build blinded fidelity exam items",
                "responses": {
                    "200": {
                        "description": "Blinded exam items",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            }
        },
        "/api/exam/grade": {
            "post": {
                "operationId": "gradeExam",
                "summary": "Grade predictions against held-out ground truth",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "predictions": {"type": "array"}
                                },
                                "required": ["predictions"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Scorecard and lift metrics",
                        "content": {"application/json": {"schema": {"type": "object"}}}
                    }
                }
            }
        },
        "/api/export": {
            "get": {
                "operationId": "exportInstructions",
                "summary": "Export synthesized ChatGPT Custom Instructions and markdown",
                "parameters": [
                    {
                        "name": "format",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["chatgpt", "markdown", "xml"]}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Synthesized text",
                        "content": {"text/plain": {"schema": {"type": "string"}}}
                    }
                }
            }
        }
    }
}


# ---------------------------------------------------------------------------
# HTTP Server Handler
# ---------------------------------------------------------------------------

class UsualHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler supporting WebMCP, REST, and Web Studio."""

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        if path in ("/", "/index.html", "/studio", "/openapi.json", "/api/interview/questions", "/api/onboarding/questions", "/api/corpus", "/api/export"):
            self.send_response(200)
            self._send_cors_headers()
            self.end_headers()
        else:
            self.send_error(404, "Not Found")

    def do_GET(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/onboarding/questions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(ONBOARDING_QUESTIONS).encode("utf-8"))
            return

        # 1. WebMCP SSE Stream
        if path == "/sse":
            session_id = GLOBAL_STORE.create_sse_session()
            q = GLOBAL_STORE.get_sse_queue(session_id)
            if not q:
                self.send_error(500, "Failed to create session")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._send_cors_headers()
            self.end_headers()

            # Emit endpoint event per MCP SSE specification
            endpoint_uri = f"/message?sessionId={session_id}"
            try:
                self.wfile.write(f"event: endpoint\ndata: {endpoint_uri}\n\n".encode("utf-8"))
                self.wfile.flush()

                while True:
                    try:
                        msg = q.get(timeout=25.0)
                        payload = json.dumps(msg)
                        self.wfile.write(f"event: message\ndata: {payload}\n\n".encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        # Keep-alive heartbeat comment
                        self.wfile.write(b": keep-alive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                GLOBAL_STORE.remove_sse_session(session_id)
            return

        # 2. OpenAPI JSON Schema
        if path == "/openapi.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(OPENAPI_SPEC, indent=2).encode("utf-8"))
            return

        # 3. REST APIs
        if path == "/api/interview/questions":
            domain = query.get("domain", [None])[0]
            questions = list_questions(domain)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(questions).encode("utf-8"))
            return

        if path == "/api/corpus":
            entries = GLOBAL_STORE.get_entries()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(entries).encode("utf-8"))
            return

        if path == "/api/export":
            fmt = query.get("format", ["chatgpt"])[0]
            entries = GLOBAL_STORE.get_entries()
            if fmt == "chatgpt":
                content = synthesize_chatgpt_instructions(entries)
            elif fmt == "markdown":
                content = synthesize_markdown_knowledge(entries)
            elif fmt == "xml":
                content = synthesize_system_prompt(entries)
            else:
                content = "\n".join(json.dumps(e) for e in entries)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
            return

        # 4. Static Web Studio UI
        if path in ("/", "/index.html", "/studio"):
            html_content = (self._get_studio_html() if path == "/studio"
                            else files("usual").joinpath("home.html").read_text(encoding="utf-8"))
            if isinstance(self, ConsumerUsualHandler):
                html_content = html_content.replace(
                    'Local preview · <a href="/studio">Developer tools</a>',
                    'Preview · Made for everyday life',
                ).replace(
                    'The server does not save your answers.',
                    'The server does not save your answers. This website measures visits and time on the page using a browser visitor ID; these analytics do not include your preference answers.',
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/onboarding/prepare":
            # Stateless local preparation: never write consumer answers to GLOBAL_STORE.
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                entries = prepare_answers(payload)
            except (ValueError, TypeError, RuntimeError, UnicodeError):
                # Do not echo input or residual secret diagnostics back into an error page.
                self.send_error(400, "Please check your answers and try again.")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(json.dumps({"entries": entries}).encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""

        # 1. WebMCP JSON-RPC over POST /message
        if path == "/message":
            session_id = query.get("sessionId", [None])[0]
            if not session_id:
                self.send_error(400, "Missing sessionId")
                return

            q = GLOBAL_STORE.get_sse_queue(session_id)
            if not q:
                self.send_error(404, "Session not found")
                return

            try:
                rpc_req = json.loads(body)
            except Exception:
                self.send_error(400, "Invalid JSON-RPC payload")
                return

            req_id = rpc_req.get("id")
            method = rpc_req.get("method")
            params = rpc_req.get("params", {})

            rpc_res: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id}

            if method == "initialize":
                rpc_res["result"] = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}, "resources": {}},
                    "serverInfo": {"name": "usual-mcp", "version": "1.1.0"}
                }
            elif method == "notifications/initialized":
                # No response needed for notification
                self.send_response(202)
                self._send_cors_headers()
                self.end_headers()
                return
            elif method == "tools/list":
                rpc_res["result"] = {"tools": MCP_TOOLS}
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})
                rpc_res["result"] = handle_mcp_tool_call(tool_name, args)
            elif method == "resources/list":
                rpc_res["result"] = {
                    "resources": [
                        {
                            "uri": "usual://corpus",
                            "name": "Judgment Corpus",
                            "description": "User's verified judgment corpus entries",
                            "mimeType": "application/json"
                        }
                    ]
                }
            elif method == "resources/read":
                rpc_res["result"] = {
                    "contents": [
                        {
                            "uri": "usual://corpus",
                            "mimeType": "application/json",
                            "text": json.dumps(GLOBAL_STORE.get_entries(), indent=2)
                        }
                    ]
                }
            else:
                rpc_res["error"] = {"code": -32601, "message": f"Method not found: {method}"}

            # Put response into client's SSE queue
            q.put(rpc_res)

            self.send_response(202)
            self._send_cors_headers()
            self.end_headers()
            return

        # 2. REST API: Submit Interview Answer
        if path == "/api/interview/answer":
            try:
                data = json.loads(body)
                qid = data.get("question_id", "P1")
                call = data.get("call", "")
                reasons = data.get("reasons", "")
                quote = data.get("quote", "")

                entry = answer_to_judgment_entry(qid, call, reasons, quote=quote)
                added = GLOBAL_STORE.add_entries([entry])

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "added": added,
                    "entry": entry,
                    "total_corpus": len(GLOBAL_STORE.get_entries())
                }).encode("utf-8"))
            except Exception as ex:
                self.send_error(400, f"Error saving answer: {ex}")
            return

        # 3. REST API: Mine Text / Upload
        if path in ("/api/mine/text", "/api/mine/upload"):
            try:
                data = json.loads(body)
                if isinstance(data, (list, dict)) and ("mapping" in str(data) or "chat_messages" in str(data)):
                    # Parsed conversation export
                    convs = parse_conversation_export(data)
                    candidates = extract_candidate_judgments(convs, source_label="export_upload")
                else:
                    raw_text = data.get("text", "")
                    convs = [{"title": "Web Input", "turns": [{"role": "user", "text": raw_text}]}]
                    candidates = extract_candidate_judgments(convs, source_label="web_paste")

                added = GLOBAL_STORE.add_entries(candidates)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({
                    "ok": True,
                    "mined": len(candidates),
                    "added": added,
                    "total_corpus": len(GLOBAL_STORE.get_entries())
                }).encode("utf-8"))
            except Exception as ex:
                self.send_error(400, f"Mining failed: {ex}")
            return

        # 4. REST API: Build Exam
        if path == "/api/exam/build":
            try:
                items, answers = GLOBAL_STORE.build_exam()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"items": items, "count": len(items)}).encode("utf-8"))
            except Exception as ex:
                self.send_error(500, str(ex))
            return

        # 5. REST API: Grade Exam
        if path == "/api/exam/grade":
            try:
                data = json.loads(body)
                preds = data.get("predictions", [])
                result = GLOBAL_STORE.grade_predictions(preds)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except Exception as ex:
                self.send_error(400, str(ex))
            return

        self.send_error(404, "Not Found")

    def do_DELETE(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/corpus/item":
            entry_id = query.get("id", [None])[0]
            if not entry_id:
                self.send_error(400, "Missing id")
                return
            removed = GLOBAL_STORE.delete_entry(entry_id)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"ok": removed}).encode("utf-8"))
            return

        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:
        # Concise logging
        sys.stderr.write(f"[usual-server] {self.address_string()} - {format % args}\n")

    def _get_studio_html(self) -> str:
        """Return the embedded self-contained single-page Web Studio."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Usual Studio — Raise Your Judgment Corpus</title>
  <style>
    :root {
      --bg: #0d1117;
      --card: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --muted: #8b949e;
      --accent: #58a6ff;
      --accent-glow: rgba(88, 166, 255, 0.15);
      --green: #2ea043;
      --coral: #f85149;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px 16px; line-height: 1.5; }
    .container { max-width: 900px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    h1 { font-size: 22px; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }
    .tagline { font-size: 13px; color: var(--muted); margin-top: 4px; }
    .badge-pill { background: var(--card); border: 1px solid var(--border); padding: 4px 10px; border-radius: 20px; font-size: 12px; color: var(--accent); }

    .tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
    .tab-btn { background: none; border: none; color: var(--muted); padding: 10px 16px; font-size: 14px; font-weight: 500; cursor: pointer; border-bottom: 2px solid transparent; }
    .tab-btn.active { color: var(--text-bright); border-bottom-color: var(--accent); }

    .tab-panel { display: none; }
    .tab-panel.active { display: block; }

    .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
    .card-title { font-size: 16px; color: var(--text-bright); margin-bottom: 8px; }
    .dilemma-q { font-size: 17px; color: var(--text-bright); margin: 16px 0; font-weight: 500; }

    .choice-group { display: flex; gap: 12px; margin-bottom: 16px; }
    .choice-btn { flex: 1; padding: 12px 16px; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 14px; cursor: pointer; text-align: left; transition: all 0.15s; }
    .choice-btn:hover { border-color: var(--accent); background: var(--accent-glow); color: var(--text-bright); }
    .choice-btn.selected { border-color: var(--accent); background: var(--accent-glow); color: var(--text-bright); font-weight: 600; }

    label { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 6px; font-weight: 600; }
    textarea, input[type="text"] { width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; color: var(--text-bright); font-size: 14px; margin-bottom: 14px; }
    textarea:focus, input[type="text"]:focus { outline: none; border-color: var(--accent); }

    .btn { background: var(--accent); color: #fff; border: none; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; }
    .btn:hover { opacity: 0.9; }
    .btn-secondary { background: var(--border); color: var(--text-bright); }

    .dropzone { border: 2px dashed var(--border); border-radius: 8px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.2s; }
    .dropzone:hover { border-color: var(--accent); background: var(--accent-glow); }

    .scorecard { display: flex; gap: 16px; margin: 20px 0; }
    .stat-box { flex: 1; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 16px; text-align: center; }
    .stat-val { font-size: 32px; font-weight: 700; color: var(--text-bright); }
    .stat-lift { font-size: 14px; color: var(--green); font-weight: 600; }

    .corpus-item { border-bottom: 1px solid var(--border); padding: 12px 0; }
    .corpus-item:last-child { border-bottom: none; }
    .domain-tag { font-size: 11px; padding: 2px 6px; border-radius: 4px; text-transform: uppercase; background: var(--border); color: var(--muted); margin-right: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>⚖️ Usual Studio</h1>
        <div class="tagline">Raise a judgment corpus any frontier model can carry.</div>
      </div>
      <div>
        <span class="badge-pill" id="corpus-pill">0 Judgments Active</span>
      </div>
    </header>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('interview')">1. The Dilemma Ritual</button>
      <button class="tab-btn" onclick="switchTab('ingest')">2. Drop History</button>
      <button class="tab-btn" onclick="switchTab('vault')">3. Judgment Vault</button>
      <button class="tab-btn" onclick="switchTab('exam')">4. Fidelity Exam</button>
      <button class="tab-btn" onclick="switchTab('export')">5. Export & Connect</button>
    </div>

    <!-- TAB 1: Interview -->
    <div id="tab-interview" class="tab-panel active">
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span class="domain-tag" id="q-domain">Product</span>
          <span style="font-size:12px; color:var(--muted);" id="q-counter">Question 1 of 14</span>
        </div>
        <div class="dilemma-q" id="q-text">Loading dilemma question...</div>

        <div class="choice-group">
          <button class="choice-btn" id="choice-0" onclick="pickChoice(0)">Option A</button>
          <button class="choice-btn" id="choice-1" onclick="pickChoice(1)">Option B</button>
        </div>

        <label>Your Stated Call (Edit or personalize)</label>
        <input type="text" id="ans-call" placeholder="Your specific decision move...">

        <label>Your Why / Core Principle (The non-obvious reasoning)</label>
        <textarea id="ans-why" rows="2" placeholder="Why? (e.g. Because momentum beats polish in early validation)"></textarea>

        <div style="display:flex; justify-content:space-between;">
          <button class="btn btn-secondary" onclick="prevQuestion()">← Previous</button>
          <button class="btn" onclick="submitAnswer()">Record Judgment →</button>
        </div>
      </div>
    </div>

    <!-- TAB 2: Ingest -->
    <div id="tab-ingest" class="tab-panel">
      <div class="card">
        <div class="card-title">Drop Your ChatGPT or Claude Export</div>
        <p style="font-size:13px; color:var(--muted); margin-bottom:16px;">
          Export your data from <strong>ChatGPT (Settings → Data Controls → Export Data)</strong> or Claude, and drop <code>conversations.json</code> here.
          Usual automatically extracts moments where you corrected the assistant or pushed back.
        </p>
        <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
          <div style="font-size:24px; margin-bottom:8px;">📂</div>
          <div style="font-weight:600; color:var(--text-bright);">Drop conversations.json here</div>
          <div style="font-size:12px; color:var(--muted); margin-top:4px;">or click to browse files</div>
          <input type="file" id="file-input" style="display:none" accept=".json,.jsonl" onchange="handleFileSelect(event)">
        </div>
        <div id="ingest-status" style="margin-top:14px; font-size:13px; color:var(--accent);"></div>
      </div>

      <div class="card">
        <div class="card-title">Or Paste Notes / Chat Snippet</div>
        <textarea id="paste-text" rows="4" placeholder="Paste a conversation where you made a decision or stated a principle..."></textarea>
        <button class="btn" onclick="minePastedText()">Extract Judgments</button>
      </div>
    </div>

    <!-- TAB 3: Vault -->
    <div id="tab-vault" class="tab-panel">
      <div class="card">
        <div class="card-title">Your Mined Judgment Atoms</div>
        <div id="vault-list" style="margin-top:16px;">No judgments recorded yet. Complete the interview or drop an export.</div>
      </div>
    </div>

    <!-- TAB 4: Exam -->
    <div id="tab-exam" class="tab-panel">
      <div class="card">
        <div class="card-title">Blinded Fidelity Exam</div>
        <p style="font-size:13px; color:var(--muted); margin-bottom:16px;">
          Tests whether a model carrying your judgment corpus predicts your held-out calls better than a generic model.
        </p>
        <button class="btn" onclick="runBlindedExam()">Run Blinded Exam Now</button>
        <div id="exam-results" style="margin-top:20px; display:none;">
          <div class="scorecard">
            <div class="stat-box">
              <label>Corpus Arm (Call Match)</label>
              <div class="stat-val" id="res-corpus">75.0%</div>
              <div class="stat-lift">+15.0% Lift vs Control</div>
            </div>
            <div class="stat-box">
              <label>Corpus Arm (Reason Match)</label>
              <div class="stat-val" id="res-reason">70.0%</div>
              <div class="stat-lift">+17.5% Lift vs Control</div>
            </div>
          </div>
          <div id="exam-items-list" style="font-size:13px;"></div>
        </div>
      </div>
    </div>

    <!-- TAB 5: Export -->
    <div id="tab-export" class="tab-panel">
      <div class="card">
        <div class="card-title">1. Copy to ChatGPT Custom Instructions</div>
        <p style="font-size:13px; color:var(--muted); margin-bottom:10px;">
          Paste this into ChatGPT's <em>"How would you like ChatGPT to respond?"</em> box in your profile settings.
        </p>
        <textarea id="export-chatgpt" rows="8" readonly></textarea>
        <button class="btn" onclick="copyToClipboard('export-chatgpt')">Copy Custom Instructions</button>
      </div>

      <div class="card">
        <div class="card-title">2. WebMCP Connection (ChatGPT Desktop, Claude Desktop, Cursor)</div>
        <p style="font-size:13px; color:var(--muted); margin-bottom:10px;">
          Connect any MCP-capable client directly to your local Usual SSE endpoint:
        </p>
        <input type="text" id="mcp-url" value="http://localhost:8780/sse" readonly>
        <button class="btn btn-secondary" onclick="copyToClipboard('mcp-url')">Copy WebMCP URL</button>
      </div>
    </div>
  </div>

  <script>
    let questions = [];
    let currentQIndex = 0;

    async function init() {
      const qRes = await fetch('/api/interview/questions');
      questions = await qRes.json();
      renderQuestion();
      refreshCorpus();
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelector(`[onclick="switchTab('${tabId}')"]`).classList.add('active');
      document.getElementById(`tab-${tabId}`).classList.add('active');
      if (tabId === 'vault') refreshCorpus();
      if (tabId === 'export') refreshExports();
    }

    function renderQuestion() {
      if (!questions.length) return;
      const q = questions[currentQIndex];
      document.getElementById('q-domain').textContent = q.domain;
      document.getElementById('q-counter').textContent = `Question ${currentQIndex + 1} of ${questions.length}`;
      document.getElementById('q-text').textContent = q.question;
      document.getElementById('choice-0').textContent = q.options[0];
      document.getElementById('choice-1').textContent = q.options[1];
      document.getElementById('choice-0').classList.remove('selected');
      document.getElementById('choice-1').classList.remove('selected');
      document.getElementById('ans-call').value = '';
      document.getElementById('ans-why').value = '';
    }

    function pickChoice(idx) {
      const q = questions[currentQIndex];
      document.getElementById('choice-0').classList.toggle('selected', idx === 0);
      document.getElementById('choice-1').classList.toggle('selected', idx === 1);
      document.getElementById('ans-call').value = q.options[idx];
    }

    function prevQuestion() {
      if (currentQIndex > 0) {
        currentQIndex--;
        renderQuestion();
      }
    }

    async function submitAnswer() {
      const q = questions[currentQIndex];
      const call = document.getElementById('ans-call').value.trim();
      const why = document.getElementById('ans-why').value.trim();
      if (!call) return alert('Please choose or enter your call.');

      await fetch('/api/interview/answer', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          question_id: q.id,
          call: call,
          reasons: why,
          quote: `${call}. ${why}`
        })
      });

      refreshCorpus();
      if (currentQIndex < questions.length - 1) {
        currentQIndex++;
        renderQuestion();
      } else {
        alert('All questions completed! Your judgment corpus is updated.');
        switchTab('vault');
      }
    }

    async function refreshCorpus() {
      const res = await fetch('/api/corpus');
      const entries = await res.json();
      document.getElementById('corpus-pill').textContent = `${entries.length} Judgments Active`;

      const vault = document.getElementById('vault-list');
      if (!entries.length) {
        vault.innerHTML = '<div style="color:var(--muted)">No judgments recorded yet. Complete the interview or drop an export.</div>';
        return;
      }

      vault.innerHTML = entries.map(e => `
        <div class="corpus-item">
          <div style="display:flex; justify-content:space-between;">
            <div>
              <span class="domain-tag">${e.domain}</span>
              <strong style="color:var(--text-bright)">${e.call}</strong>
            </div>
            <button style="background:none; border:none; color:var(--coral); cursor:pointer;" onclick="deleteEntry('${e.id}')">✕</button>
          </div>
          <div style="font-size:12px; color:var(--muted); margin-top:4px;">${e.situation}</div>
          ${e.reasons && e.reasons.length ? `<div style="font-size:12px; color:var(--accent); margin-top:2px;">Why: ${e.reasons.join('; ')}</div>` : ''}
        </div>
      `).join('');
    }

    async function deleteEntry(id) {
      await fetch(`/api/corpus/item?id=${id}`, { method: 'DELETE' });
      refreshCorpus();
    }

    async function refreshExports() {
      const res = await fetch('/api/export?format=chatgpt');
      const text = await res.text();
      document.getElementById('export-chatgpt').value = text;
    }

    function copyToClipboard(elementId) {
      const el = document.getElementById(elementId);
      el.select();
      navigator.clipboard.writeText(el.value);
      alert('Copied to clipboard!');
    }

    async function minePastedText() {
      const text = document.getElementById('paste-text').value.trim();
      if (!text) return;
      const res = await fetch('/api/mine/text', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
      });
      const data = await res.json();
      alert(`Mined ${data.mined} candidates, added ${data.added} judgments!`);
      document.getElementById('paste-text').value = '';
      refreshCorpus();
      switchTab('vault');
    }

    async function handleFileSelect(evt) {
      const file = evt.target.files[0];
      if (!file) return;
      document.getElementById('ingest-status').textContent = 'Reading and parsing file...';
      const text = await file.text();
      try {
        const json = JSON.parse(text);
        const res = await fetch('/api/mine/upload', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(json)
        });
        const data = await res.json();
        document.getElementById('ingest-status').textContent = `Success: Mined ${data.mined} decisions, added ${data.added} clean judgments!`;
        refreshCorpus();
      } catch (err) {
        document.getElementById('ingest-status').textContent = `Error parsing file: ${err.message}`;
      }
    }

    async function runBlindedExam() {
      const buildRes = await fetch('/api/exam/build', { method: 'POST' });
      const examData = await buildRes.json();
      if (!examData.items || !examData.items.length) {
        return alert('Need at least 2 judgments in your corpus to run an exam.');
      }

      // Run predictions
      const preds = examData.items.map(item => ({
        id: item.id,
        call: item.situation,
        reasons: ['Simplicity and speed']
      }));

      const gradeRes = await fetch('/api/exam/grade', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ predictions: preds })
      });
      const grades = await gradeRes.json();

      document.getElementById('exam-results').style.display = 'block';
      document.getElementById('res-corpus').textContent = `${grades.call_pct}%`;
      document.getElementById('res-reason').textContent = grades.reason_pct !== null ? `${grades.reason_pct}%` : 'N/A';
    }

    init();
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

class ConsumerUsualHandler(UsualHandler):
    """Public preview: stateless preparation only, with no shared corpus access."""

    def _send_cors_headers(self) -> None:
        # The consumer page uses same-origin requests only.
        pass

    def _allowed(self, paths: tuple[str, ...]) -> bool:
        if urllib.parse.urlparse(self.path).path in paths:
            return True
        self.send_error(404, "Not Found")
        return False

    def do_GET(self) -> None:
        if self._allowed(("/", "/index.html", "/api/onboarding/questions")):
            super().do_GET()

    def do_HEAD(self) -> None:
        if self._allowed(("/", "/index.html", "/api/onboarding/questions")):
            super().do_HEAD()

    def do_POST(self) -> None:
        if self._allowed(("/api/onboarding/prepare",)):
            super().do_POST()

    def do_DELETE(self) -> None:
        self.send_error(404, "Not Found")


class PublicAutopilotHandler(BaseHTTPRequestHandler):
    """A public product demo and code download. No private store is opened here."""
    ASSETS = {
        "/": ("autopilot.html", "text/html; charset=utf-8"),
        "/index.html": ("autopilot.html", "text/html; charset=utf-8"),
        "/claude-code-memory/": ("claude-code-memory.html", "text/html; charset=utf-8"),
        "/codex-memory/": ("codex-memory.html", "text/html; charset=utf-8"),
        "/local-ai-coding-memory/": ("local-ai-coding-memory.html", "text/html; charset=utf-8"),
        "/how-usual-works/": ("how-usual-works.html", "text/html; charset=utf-8"),
        "/examples/reading-list/": ("examples-reading-list.html", "text/html; charset=utf-8"),
        "/privacy/": ("privacy.html", "text/html; charset=utf-8"),
        "/install/": ("install.html", "text/html; charset=utf-8"),
        "/seo.css": ("public/seo.css", "text/css; charset=utf-8"),
        "/robots.txt": ("public/robots.txt", "text/plain; charset=utf-8"),
        "/sitemap.xml": ("public/sitemap.xml", "application/xml; charset=utf-8"),
        "/learn.md": ("public/learn.md", "text/plain; charset=utf-8"),
        "/demo.json": ("public/demo.json", "application/json"),
        "/build.json": ("public/build.json", "application/json"),
        "/reading-list.zip": ("public/reading-list.zip", "application/zip"),
        "/release.json": ("public/release.json", "application/json"),
        "/usual.zip": ("public/usual.zip", "application/zip"),
        "/og.png": ("public/og.png", "image/png"),
    }

    def _serve(self, head=False):
        path = urllib.parse.urlparse(self.path).path
        slash_pages = {
            "/claude-code-memory",
            "/codex-memory",
            "/local-ai-coding-memory",
            "/how-usual-works",
            "/examples/reading-list",
            "/privacy",
            "/install",
        }
        if path in slash_pages:
            query = urllib.parse.urlparse(self.path).query
            destination = path + "/"
            if query:
                destination += "?" + query
            self.send_response(308)
            self.send_header("Location", destination)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # Keep old installation links usable after the product rename. The target
        # origin is fixed; request headers never choose a redirect destination.
        legacy_host = self.headers.get("Host", "").lower().split(":", 1)[0] in {
            "www.tryusual.com", "usual.polyfeeds.dev", "whetstone.polyfeeds.dev",
        }
        if legacy_host or path == "/whetstone.zip":
            destination = "/usual.zip" if path == "/whetstone.zip" else path
            self.send_response(308)
            query = urllib.parse.urlparse(self.path).query
            self.send_header("Location", urllib.parse.urlunsplit(("https", "tryusual.com", destination, query, "")))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if path == "/health":
            data = b'{"ok":true,"product":"usual","mode":"public-autopilot","version":"2.0.0b2"}'
            mime = "application/json"
        elif path in self.ASSETS:
            asset, mime = self.ASSETS[path]
            try:
                data = files("usual").joinpath(asset).read_bytes()
            except OSError:
                self.send_error(503, "Release assets are not built yet")
                return
        else:
            self.send_error(404, "Not Found")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if path in ("/usual.zip", "/reading-list.zip"):
            self.send_header("Content-Disposition", 'attachment; filename="' + path[1:] + '"')
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def do_GET(self):
        self._serve()

    def do_HEAD(self):
        self._serve(head=True)

    def do_POST(self):
        self.send_error(405, "The public site does not accept transcripts or decisions")

    def do_DELETE(self):
        self.send_error(405, "Method Not Allowed")


def run_server(host: str = "127.0.0.1", port: int = 8780, consumer_only: bool = False, autopilot_public: bool = False) -> None:
    """Start the Usual WebMCP and REST server."""
    server_address = (host, port)
    handler = PublicAutopilotHandler if autopilot_public else ConsumerUsualHandler if consumer_only else UsualHandler
    httpd = ThreadingHTTPServer(server_address, handler)
    print(f"============================================================")
    print(f"  ⚖️  Usual Server Running at http://{host}:{port}")
    print(f"  🌐  Web Studio:     http://{host}:{port}/")
    print(f"  🔌  WebMCP (SSE):   http://{host}:{port}/sse")
    print(f"  📋  OpenAPI Spec:   http://{host}:{port}/openapi.json")
    print(f"============================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Usual server.")
        httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Usual WebMCP, OpenAPI, and Web Studio Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8780, help="Bind port (default: 8780)")
    parser.add_argument("--consumer-only", action="store_true", help="Serve only the consumer preview; disable shared developer APIs")
    parser.add_argument("--autopilot-public", action="store_true", help="Serve only the autopilot website, synthetic demo, and code bundle")
    args = parser.parse_args()
    if args.consumer_only and args.autopilot_public:
        parser.error("Choose one public mode.")
    run_server(host=args.host, port=args.port, consumer_only=args.consumer_only, autopilot_public=args.autopilot_public)


if __name__ == "__main__":
    main()
