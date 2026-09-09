"""Streaming decision-episode mining from native Codex and Claude histories.

Links observed human answers to questions; never infers universal preferences.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .transcripts import content_text, human_text, scrub

MAX_LINE = 16 * 1024 * 1024
MAX_CONTEXT = 12000


def history_files(provider="both", *, include_archived=True):
    roots = []
    if provider in ("both", "codex"):
        roots.append(("codex", Path.home()/".codex/sessions"))
        if include_archived:
            roots.append(("codex", Path.home()/".codex/archived_sessions"))
    if provider in ("both", "claude"):
        roots.append(("claude", Path.home()/".claude/projects"))
    files, excluded = [], Counter()
    for source, root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            if path.is_symlink():
                excluded["symlink_files"] += 1
            elif "subagents" in path.parts:
                excluded["subagent_files"] += 1
            else:
                files.append((source, path))
    return sorted(files, key=lambda item: str(item[1])), dict(excluded)


def _questions(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return []
    if not isinstance(arguments, dict):
        return []
    result = []
    for index, item in enumerate(arguments.get("questions", [])):
        if not isinstance(item, dict):
            continue
        question = item.get("question", item.get("title", ""))
        if not isinstance(question, str) or not question.strip():
            continue
        options = []
        for option in item.get("options", []):
            if isinstance(option, str):
                options.append({"label": option})
            elif isinstance(option, dict) and isinstance(option.get("label"), str):
                options.append({"label": option["label"], "description": str(option.get("description", ""))})
        result.append({"question": question, "options": options, "id": item.get("id", question), "index": index})
    return result


def _result_answers(value, questions):
    """Decode a linked question tool's answer format, not arbitrary tool output."""
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except ValueError:
            decoded = None
        if isinstance(decoded, dict):
            value = decoded
    if isinstance(value, dict):
        answers = value.get("answers", {})
        if isinstance(answers, dict):
            for question in questions:
                response = answers.get(question["id"], answers.get(question["question"]))
                if isinstance(response, dict):
                    response = response.get("answers", response.get("answer"))
                if isinstance(response, list) and all(isinstance(a, str) for a in response):
                    response = ", ".join(response)
                if isinstance(response, str) and response.strip():
                    yield question, response
        return
    if not isinstance(value, str) or not value.startswith(("The user answered:", "User has answered your questions:")):
        return
    # Claude serializes literal quotes inside questions; match the actual known question.
    starts = []
    for question in questions:
        marker = '"' + question["question"] + '"="'
        position = value.find(marker)
        if position >= 0:
            starts.append((position, position + len(marker), question))
    starts.sort(key=lambda row: row[0])
    for i, (_, beginning, question) in enumerate(starts):
        end = starts[i+1][0] if i+1 < len(starts) else len(value)
        segment = value[beginning:end]
        if i+1 < len(starts):
            answer = segment.removesuffix(', ').removesuffix('"')
        else:
            markers = ['". Read the answers', '". You can now', '". You should now']
            boundary = next((segment.find(m) for m in markers if m in segment), -1)
            answer = segment[:boundary] if boundary >= 0 else segment.rstrip().removesuffix('"')
        if answer.strip():
            yield question, answer


def question_prompt(value):
    prose = re.sub(r"```[\s\S]*?```|`[^`]*`|“[^”]*”|\"[^\"\n]*\"", "", value)
    prose = "\n".join(line for line in prose.splitlines() if not line.lstrip().startswith(">"))
    if prose.strip().endswith("?"):
        return True
    return bool(re.search(r"\b(?:would you|do you|should i|should we|shall i|can i|may i|want me|which|what would|what should|what do you|how should|are you|could you|can you)\b[^?\n]*\?", prose, re.I))


def mine_episodes(path, provider):
    """Yield redacted episodes; final item carries counts. Memory is bounded per line."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Transcript symlinks are not imported.")
    stats = Counter()
    session, project, previous = path.stem, "", None
    pending, awaiting = {}, []
    last_user_signature = None

    def make(question, answer, line, date, link, question_line):
        nonlocal last_user_signature
        answer = human_text(answer)
        if not answer:
            stats["ignored_answers"] += 1
            return None
        # Retain short replies. Their question provides the meaning.
        question_text = question["question"]
        truncated = len(question_text) > MAX_CONTEXT or len(answer) > MAX_CONTEXT
        question_text, answer = scrub(question_text[:MAX_CONTEXT]), scrub(answer[:MAX_CONTEXT])
        options = [{k: scrub(str(v)[:4000]) for k, v in o.items()} for o in question.get("options", [])]
        identity = "\0".join([provider, session, str(question_line), str(line), question_text, answer])
        digest = hashlib.sha256(identity.encode()).hexdigest()
        normalized_answer = " ".join(answer.casefold().split())
        selected = [o["label"] for o in options if " ".join(o["label"].casefold().split()) == normalized_answer]
        return {"id": "ep_" + digest[:24], "provider": provider, "session": scrub(session),
                "project": scrub(project), "source": scrub(str(path)), "question_line": question_line,
                "answer_line": line, "date": date, "question": question_text, "options": options,
                "answer": answer, "selected_options": selected, "link": link,
                "assistant_followup": "", "context_truncated": truncated}

    def user_message(value, line, date):
        nonlocal previous, last_user_signature
        # Async Codex replies carry a call ID + question index. Do not attach them to
        # whichever assistant prose happened to arrive while the question was pending.
        match = re.search(r"<send_user_message_question_reply>\s*([\s\S]*?)\s*</send_user_message_question_reply>", value)
        if match:
            try:
                responses = json.loads(match.group(1))
                for response in responses:
                    identity = json.loads(response["questionItemId"])
                    call_id, index = identity[1], identity[2]
                    item = pending.get(call_id)
                    if not item:
                        stats["unmatched_tool_answers"] += 1
                        continue
                    question = next((q for q in item["questions"] if q["index"] == index), None)
                    if question and question["index"] in item["answered"]:
                        stats["duplicate_tool_answers"] += 1
                        continue
                    if question and isinstance(response.get("answer"), str):
                        item["answered"].add(question["index"])
                        episode = make(question, response["answer"], line, date, "tool_linked", item["line"])
                        if episode:
                            awaiting.append(episode)
                    else:
                        stats["unmatched_tool_answers"] += 1
            except (ValueError, KeyError, TypeError, IndexError):
                stats["unmatched_tool_answers"] += 1
            previous = None
            return
        cleaned = human_text(value)
        signature = hashlib.sha256(cleaned.encode()).hexdigest()
        if signature == last_user_signature and previous is None:
            stats["duplicate_user_events"] += 1
            return
        last_user_signature = signature
        if previous and cleaned:
            episode = make({"question": previous["text"], "options": []}, cleaned, line, date,
                           "adjacent_reply", previous["line"])
            if episode:
                awaiting.append(episode)
        previous = None  # ignored user messages are barriers too

    with path.open("rb") as stream:
        line_no = 0
        while True:
            data = stream.readline(MAX_LINE + 1)
            if not data:
                break
            line_no += 1
            if len(data) > MAX_LINE:
                while data and not data.endswith(b"\n"):
                    data = stream.readline(MAX_LINE + 1)
                stats["oversized_lines"] += 1
                previous = None
                continue
            try:
                record = json.loads(data)
                if not isinstance(record, dict):
                    raise ValueError()
            except (ValueError, UnicodeError):
                stats["malformed_lines"] += 1
                previous = None
                continue
            stats["records"] += 1
            if record.get("isSidechain") or record.get("isMeta"):
                stats["ignored_sidechain_or_meta"] += 1
                previous = None
                continue
            kind = record.get("type")
            payload = record.get("payload", {})
            payload = payload if isinstance(payload, dict) else {}
            date = str(record.get("timestamp", ""))[:10]
            if kind == "session_meta":
                source = payload.get("source")
                if isinstance(source, dict) and "subagent" in source:
                    stats["excluded_subagent_session"] += 1
                    break
                project = str(payload.get("cwd", project))
                session = str(payload.get("id", session))
            if kind == "turn_context":
                project = str(payload.get("cwd", project))
            messages, calls, results = [], [], []
            if kind == "response_item":
                ptype = payload.get("type")
                if ptype == "message" and payload.get("role") in ("assistant", "user") and payload.get("channel") not in ("analysis", "summary"):
                    messages.append((payload["role"], content_text(payload.get("content"))))
                elif ptype == "function_call" and str(payload.get("name", "")).split(".")[-1] in ("request_user_input", "request_user_input_async"):
                    calls.append((payload.get("call_id"), payload.get("arguments")))
                elif ptype == "function_call_output":
                    results.append((payload.get("call_id"), payload.get("output"), False))
            elif kind in ("assistant", "user"):
                project = str(record.get("cwd", project))
                message = record.get("message", {})
                if isinstance(message, dict):
                    blocks = message.get("content", [])
                    if isinstance(blocks, list):
                        for block in blocks:
                            if not isinstance(block, dict):
                                continue
                            if block.get("type") == "tool_use" and block.get("name") == "AskUserQuestion":
                                calls.append((block.get("id"), block.get("input")))
                            elif block.get("type") == "tool_result":
                                results.append((block.get("tool_use_id"), content_text(block.get("content")), block.get("is_error", False)))
                    text = content_text(blocks)
                    if text:
                        messages.append((message.get("role", kind), text))
            elif record.get("role") in ("assistant", "user"):
                messages.append((record["role"], content_text(record.get("content", record.get("text")))))
            # Codex event_msg duplicates are deliberately not processed when native
            # response_item messages exist; count event-only formats for coverage.
            elif kind == "event_msg" and payload.get("type") == "user_message":
                stats["codex_user_events_not_used"] += 1
            for call_id, arguments in calls:
                questions = _questions(arguments)
                if call_id and questions:
                    pending[call_id] = {"questions": questions, "line": line_no, "answered": set()}
                    stats["question_tools"] += 1
                    previous = None
            for call_id, value, failed in results:
                item = pending.get(call_id)
                if item and not failed:
                    for question, answer in _result_answers(value, item["questions"]):
                        if question["index"] in item["answered"]:
                            stats["duplicate_tool_answers"] += 1
                            continue
                        item["answered"].add(question["index"])
                        episode = make(question, answer, line_no, date, "tool_linked", item["line"])
                        if episode:
                            awaiting.append(episode)
                elif item and failed:
                    stats["cancelled_question_tools"] += 1
            for role, value in messages:
                if role == "assistant":
                    for episode in awaiting:
                        episode["assistant_followup"] = scrub(value[:4000])
                        stats["episodes"] += 1
                        yield episode
                    awaiting.clear()
                    previous = {"text": value, "line": line_no} if question_prompt(value) else None
                elif role == "user":
                    user_message(value, line_no, date)
            # Bound dangling async tools for huge sessions, keeping the most recent ones.
            while len(pending) > 256:
                pending.pop(next(iter(pending)))
                stats["expired_question_links"] += 1
    stats["unanswered_questions"] = sum(len(item["questions"]) - len(item["answered"]) for item in pending.values())
    for episode in awaiting:
        stats["episodes"] += 1
        yield episode
    yield {"_stats": dict(stats)}
