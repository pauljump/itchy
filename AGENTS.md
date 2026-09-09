# Usual project contract

- Product: **Usual**. Canonical repository: `https://github.com/pauljump/itchy`.
- Canonical local root: `/Users/mini-home/projects/itchy`.
- `src/usual/` is the runtime; `SKILL.md` and `install.py` are the portable agent skill.
- Run `python3 -m pytest -q` from the root. Tests discover only the active product and its demo, not archived research.
- Build public downloads with `python3 scripts/build_release.py`. The allowlist excludes private data and research artifacts.
- Public deployment is declared in `deploy/web.json`; use the control-plane fleet registry and vault runner.
- `archive/` preserves the original Itchy research and intermediate experiments. These are historical sources, not current product instructions or validated Usual accuracy results.
- Keep old human decision records intact during migrations. Runtime data belongs outside this repository.

See [README.md](README.md), [runtime semantics](references/runtime.md), and [migration](references/migration.md).
