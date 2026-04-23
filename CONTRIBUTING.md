# Contributing to PIE Workbench

Thanks for your interest in contributing. PIE Workbench is maintained by a
small team, so a little up-front coordination goes a long way.

## Before you start

- **Open an issue first** for anything non-trivial (new features, UI
  rewrites, dependency changes). A short discussion avoids wasted work.
- **Small fixes** (typos, docs, obvious bugs) can go straight to a PR.
- **Security issues**: please follow `SECURITY.md` — don't file a public issue.

## Development setup

See the `Installation` and `Running the Application` sections of `README.md`.
The short version:

```bash
git clone --recurse-submodules https://github.com/MJFF-ResearchCommunity/PIE-Workbench.git
cd PIE-Workbench
npm install
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -e ../lib/PIE ../lib/PIE-clean
```

Run the backend on port 8100 and the Vite dev server on 5173.

## Running tests

Backend smoke tests live under `backend/tests/`:

```bash
cd backend && source venv/bin/activate
pytest tests/ -v
```

These run entirely on synthetic fixtures — no PPMI data required. GitHub
Actions runs them on every push and PR.

## Style

- **Python**: follow the style of the surrounding code. Type hints where
  they add clarity; don't retrofit them everywhere.
- **TypeScript/React**: functional components, hooks, Tailwind for styling.
  Keep components focused; factor out reusable pieces into `src/components/`.
- **Commits**: use short, imperative subjects (`fix(stats): handle NaN in
  JSON response`). Group related changes into a single commit.

## Pull requests

1. Fork the repo and create a branch off `main`.
2. Keep the PR focused on one concern; split unrelated changes.
3. Make sure the existing tests still pass. Add tests for new backend
   endpoints; we don't currently require UI tests.
4. Describe what changed and why. Screenshots or short screen captures help
   for UI changes.

## Scope of the project

PIE Workbench is a GUI for the PIE and PIE-clean libraries, focused on
clinical researchers working with PPMI-style data. Contributions that extend
that scope are welcome (new data loaders via the `AbstractDataLoader`
pattern, new statistical tests, new visualizations). Contributions that
replace the core ML engine or pull the project away from that focus should
be discussed in an issue first.

## License

By contributing, you agree that your contributions will be licensed under
the MIT License, consistent with the rest of the project.
