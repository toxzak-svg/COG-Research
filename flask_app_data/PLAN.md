# Flask Repo Data Collection Plan

## Objective
Collect 100+ Flask repositories from GitHub with CI/CD history to train the world-model on real app evolution data.

## Data Collection Pipeline

### Phase 1: Repository Discovery (Week 1)
1. **Search GitHub API** for Flask repos with:
   - Has `app.py` or `main.py`
   - Has `requirements.txt`
   - Has GitHub Actions workflow (`.github/workflows/`)
   - Has database migrations (`alembic.ini` or `migrations/` folder)
   - Stars > 10 (quality filter)

2. **Filter for schema migrations**:
   - Look for SQLAlchemy models
   - Look for Alembic/AutoMigrate usage
   - Look for test files

### Phase 2: Data Extraction (Week 2)
For each repo, extract:
1. **Commits with schema changes**: Parse diffs for model.py changes
2. **Migration files**: Extract column/table additions
3. **Test outcomes**: Parse CI/CD logs for test pass/fail
4. **Edit sequences**: Build (state_before, edit, state_after, outcome) tuples

### Phase 3: State Encoding (Week 3)
Convert to vector representation:
- Schema: tables, columns, indices, relationships (100D)
- Endpoints: paths, methods, auth requirements (50D)
- Tests: test count, coverage, pass rate (20D)
- Total: 170D → compressed to 16D by VAE

## File Structure

```
flask_app_data/
├── collectors/
│   ├── github_collector.py     # GitHub API client
│   ├── repo_miner.py            # Clone and analyze repos
│   ├── migration_parser.py      # Parse Alembic/AutoMigrate
│   └── test_outcome_extractor.py # Parse CI/CD logs
├── dataset/
│   ├── app_state.py             # AppState dataclass
│   ├── edit_record.py           # EditRecord dataclass
│   └── dataset_builder.py       # Build train/val/test splits
├── models/
│   ├── app_encoder.py           # VAE encoder (170D → 16D)
│   └── app_world_model.py       # Transition model
└── train_app_world_model.py     # Training script

## Target Metrics
- 100 Flask repos with migrations
- 1000+ schema change events
- Target world-model accuracy: ≥75% on test outcomes

## Implementation Order
1. github_collector.py - Search and filter repos
2. repo_miner.py - Clone and extract code changes
3. app_state.py + edit_record.py - Data schemas
4. dataset_builder.py - Create training dataset
5. app_encoder.py + app_world_model.py - Model definitions
6. train_app_world_model.py - Training script

