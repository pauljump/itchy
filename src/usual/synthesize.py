"""synthesize.py — Synthesize a judgment corpus into portable system prompts and instructions.

Compiles a collection of judgment entries into:
  1. ChatGPT Custom Instructions (dense, bulleted, <= 1500 chars)
  2. Markdown Knowledge Document (rich casebook for Custom GPTs / Projects)
  3. Structured System Prompt Block (<judgment_corpus>)
"""

from __future__ import annotations

from typing import Any


def clean_rule_text(text: str) -> str:
    """Normalize text into a concise rule sentence."""
    t = text.strip()
    if not t:
        return ""
    # Capitalize first letter
    t = t[0].upper() + t[1:]
    if not t.endswith((".", "!", "?")):
        t += "."
    return t


def synthesize_chatgpt_instructions(entries: list[dict[str, Any]], max_chars: int = 1450) -> str:
    """
    Synthesize entries into dense personal decision principles tailored for
    ChatGPT's 'How would you like ChatGPT to respond?' box (1500 char limit).
    """
    if not entries:
        return "# Personal Decision Principles\n- Default to direct, unhedged answers with reasoning shown upfront."

    # Group by domain
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        d = e.get("domain", "other")
        by_domain.setdefault(d, []).append(e)

    header = "# How to Decide & Respond (Personal Alignment)\n"
    footer = "\nNever give generic 'on one hand / on the other hand' advice. State the hard tradeoff and commit."

    budget = max_chars - len(header) - len(footer)
    lines: list[str] = []
    seen_calls: set[str] = set()

    # Prioritize domains in order of impact
    domain_order = ["product", "factory", "design", "money", "people", "voice", "other"]
    sorted_domains = sorted(by_domain.keys(), key=lambda d: domain_order.index(d) if d in domain_order else 99)

    for domain in sorted_domains:
        domain_entries = by_domain[domain]
        # Sort by confidence descending
        domain_entries.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)

        for entry in domain_entries:
            call = clean_rule_text(entry.get("call", ""))
            reasons = entry.get("reasons", [])

            # Dedup similar calls
            call_key = call.lower()[:30]
            if call_key in seen_calls:
                continue
            seen_calls.add(call_key)

            if reasons:
                first_reason = reasons[0].strip()
                if not first_reason.lower().startswith("because") and not first_reason.lower().startswith("since"):
                    reason_clause = f" (Why: {first_reason})"
                else:
                    reason_clause = f" ({first_reason})"
            else:
                reason_clause = ""

            line = f"- [{domain.title()}] {call}{reason_clause}"
            if len("\n".join(lines + [line])) > budget:
                break
            lines.append(line)

        if len("\n".join(lines)) > budget - 50:
            break

    body = "\n".join(lines)
    return f"{header}{body}{footer}"


def synthesize_markdown_knowledge(entries: list[dict[str, Any]], title: str = "Personal Judgment Corpus") -> str:
    """
    Generate a full markdown casebook document suitable for uploading as
    knowledge to ChatGPT Projects, Custom GPTs, or Claude Projects.
    """
    out: list[str] = [
        f"# {title}",
        "",
        "This document contains a verified ledger of actual decisions and stated rationale.",
        "When advising or generating solutions, use these grounded heuristics rather than generic middle-ground defaults.",
        "",
    ]

    by_domain: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        d = e.get("domain", "other")
        by_domain.setdefault(d, []).append(e)

    for domain, items in sorted(by_domain.items()):
        out.append(f"## Domain: {domain.upper()}")
        out.append("")
        for i, item in enumerate(items, 1):
            sit = item.get("situation", "General context")
            call = item.get("call", "")
            reasons = item.get("reasons", [])
            date = item.get("date", "")
            prov = item.get("provenance", {}).get("quote", "")

            out.append(f"### {domain.title()} Case {i}: {call[:60]}...")
            out.append(f"- **Situation:** {sit}")
            out.append(f"- **Decision / Call:** {call}")
            if reasons:
                out.append("- **Core Reasons:**")
                for r in reasons:
                    out.append(f"  * {r}")
            if prov:
                out.append(f"- **Ground Truth Quote:** *\"{prov}\"*")
            if date:
                out.append(f"- **Date:** {date}")
            out.append("")

    return "\n".join(out)


def synthesize_system_prompt(entries: list[dict[str, Any]]) -> str:
    """
    Generate an XML-tagged <judgment_corpus> prompt block that can be prepended
    or injected into any LLM's system instructions.
    """
    lines = ["<judgment_corpus>"]
    lines.append("  <instructions>")
    lines.append("    You are equipped with the user's personal judgment corpus.")
    lines.append("    When facing ambiguity or tradeoffs, align with their historical calls and reasoning.")
    lines.append("  </instructions>")

    for i, e in enumerate(entries, 1):
        domain = e.get("domain", "other")
        sit = e.get("situation", "")
        call = e.get("call", "")
        reasons = "; ".join(e.get("reasons", []))

        lines.append(f'  <judgment id="{i}" domain="{domain}">')
        lines.append(f"    <situation>{sit}</situation>")
        lines.append(f"    <decision>{call}</decision>")
        if reasons:
            lines.append(f"    <reasons>{reasons}</reasons>")
        lines.append("  </judgment>")

    lines.append("</judgment_corpus>")
    return "\n".join(lines)
