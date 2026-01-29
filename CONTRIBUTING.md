# Contributing to HiMAP

Thanks for your interest in contributing to HiMAP (Hidden Markov models for Advanced Prognostics)!

This project is maintained by a small team, so clear issues and well-scoped pull requests help a lot.

## Ways to help

- Report bugs and provide minimal reproduction steps
- Improve documentation / examples
- Fix typos, add tests, refactor for clarity
- Propose enhancements (please open an issue first for larger changes)
---

## Quick links

- Issues: use GitHub Issues for bugs and feature requests
- Pull requests: open PRs against the default branch (or the current development/review branch if requested)
---

## Development setup

### 1) Create an environment

Use your preferred tool (conda/venv). Example:

~~~bash
python -m venv .venv
# activate it, then:
python -m pip install -U pip
~~~

### 2) Install in editable mode

Editable install makes `import himap` resolve to your working tree:

~~~bash
pip install -e .
~~~

### 3) Install pytest

~~~bash
pip install pytest
~~~
---

## About the Cython extension (HSMM)

HiMAP contains a compiled extension used by HSMM routines. Some tests (especially HSMM “smoke/integration” tests) may require a successful build.

Common build command for local development:

~~~bash
python setup_cython.py build_ext --inplace
~~~

If you cannot build the extension on your platform, you can still contribute to:
- documentation,
- utilities,
- HMM-related code,
- tests that do not require the extension.
---

## Running tests (required before opening a PR)

We currently do not enforce tests via CI, so please run them locally.

From the repository root:

~~~bash
python -m pytest
~~~


### Test expectations

Before opening a PR, please ensure:
- all tests pass locally
- any new behavior has at least one new test or updated test coverage
- changes do not break the public example/quick-start workflow
---

## Contributing workflow

### 1) Create a branch

~~~bash
git checkout -b feature/my-change
~~~

### 2) Keep PRs focused

Small, focused PRs are easier to review:
- one bug fix or one enhancement per PR if possible
- avoid mixing refactors with behavior changes unless necessary

### 3) Add/adjust tests

If you fix a bug, add a regression test that would fail without your fix.

If you add a feature, add at least:
- a unit test for core behavior, and/or
- a small integration/smoke test verifying the pipeline produces finite/sane outputs

### 4) Open a pull request

In the PR description, include:
- what changed and why
- how you tested (commands + platform)
- any performance implications (if relevant)
---

## Reporting bugs

When filing an issue, please include:
- HiMAP version (or commit hash if from source)
- Python version and OS
- a minimal snippet or steps to reproduce
- full traceback (if applicable)
- expected vs actual behavior

---

## Suggesting enhancements

For non-trivial enhancements:
1) Open an issue describing the proposal and motivation
2) Outline expected API changes (if any)
3) Discuss testing approach
4) Then implement via PR

---

## Commit Messages

- Use the present tense  
- Keep messages concise and descriptive  
- Reference related issues where relevant  

---

## Questions

If something is unclear, you are welcome to open an issue for discussion.

---

## Licensing

By submitting a contribution, you agree that your contribution can be redistributed under the project license (Apache-2.0), consistent with the repository’s licensing.
