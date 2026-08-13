# Release

Releases are published manually from GitHub Actions.

## Prepare

Update and commit the version in both places:

- `pyproject.toml`
- `doc_page_extractor/__init__.py`

The release workflow reads `pyproject.toml` and publishes tag `vX.Y.Z`.

## Publish

1. Open GitHub Actions.
2. Select the `Release` workflow.
3. Click `Run workflow` on `main`.

The workflow will:

1. Read `project.version` from `pyproject.toml`.
2. Reject the release if tag `vX.Y.Z` already exists.
3. Run tests and lint.
4. Build the package.
5. Publish to PyPI through Trusted Publishing.
6. Push tag `vX.Y.Z`.
7. Create a GitHub Release with generated notes.

## PyPI Setup

PyPI Trusted Publishing must match:

- Repository: `Moskize91/doc-page-extractor`
- Workflow: `release.yaml`
- Environment: `pypi`

No PyPI API token is required in GitHub Secrets.
