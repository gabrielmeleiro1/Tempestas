# Artifact Policy

This repository should keep source code, documentation, tests, and small fixtures in git.

Generated or heavy files should live outside normal git history:

- raw and processed full-size datasets
- trained Keras/XGBoost models
- fitted scalers
- prediction exports
- experiment logs and tuning result dumps

Recommended locations:

- keep local working artifacts under ignored data/output directories
- publish reproducible datasets and model files through DVC, Git LFS, cloud storage, or release assets
- keep tiny test inputs under `tests/fixtures/` when automated tests need CSV data

The current `.gitignore` prevents new generated artifacts from being added accidentally. Existing tracked data/model files are not removed by that rule; migrate them deliberately in a separate change if the project adopts DVC or Git LFS.
