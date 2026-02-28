"""
Realistic Flask App Training Data Generator

Generates training data based on actual Flask patterns from:
- https://flask.palletsprojects.com/en/stable/patterns/
- Common Flask extensions and best practices
"""

import json
import numpy as np
from typing import List, Dict
from pathlib import Path


# Actual Flask patterns from flask.palletsprojects.com/en/stable/patterns/
FLASK_PATTERNS = {
    'app_factory': {
        'tables': ['db', 'migrate'],
        'adds': ['create_app()', 'extensions init'],
    },
    'sqlalchemy': {
        'tables': ['User', 'Role', 'user_roles'],
        'adds': ['SQLAlchemy model', 'Column', 'relationship'],
    },
    'wtforms': {
        'tables': [],
        'adds': ['Form class', 'StringField', 'IntegerField', 'ValidationError'],
    },
    'authentication': {
        'tables': ['User', 'Session', 'login_attempts'],
        'adds': ['password_hash', 'login_user', 'logout_user', 'current_user'],
    },
    'file_uploads': {
        'tables': ['uploaded_file'],
        'adds': ['FileField', 'secure_filename', 'upload folder config'],
    },
    'caching': {
        'tables': [],
        'adds': ['cache', 'cached', 'Cache'],
    },
    'celery': {
        'tables': ['task'],
        'adds': ['Celery', 'task decorator', 'background task'],
    },
    'api': {
        'tables': ['api_key'],
        'adds': ['@app.route', 'REST endpoints', 'JSON response'],
    },
    'blueprints': {
        'tables': [],
        'adds': ['Blueprint', 'register_blueprint'],
    },
    'admin': {
        'tables': ['AdminUser'],
        'adds': ['Admin interface', 'admin view'],
    },
    'websocket': {
        'tables': ['connection'],
        'adds': ['socketio', 'emit', 'on_message'],
    },
    'testing': {
        'tables': [],
        'adds': ['test_client', 'pytest', 'fixtures'],
    },
}

# Common Flask app tables
COMMON_TABLES = [
    'user', 'post', 'comment', 'role', 'permission',
    'category', 'tag', 'post_tag', 'followers',
    'notification', 'message', 'session', 'api_key',
    'upload', 'task', 'log', 'config',
]

# Based on SQLAlchemy patterns
COMMON_COLUMNS = {
    'user': ['id', 'username', 'email', 'password_hash', 'created_at', 'updated_at', 'last_login', 'is_active', 'is_admin'],
    'post': ['id', 'user_id', 'title', 'body', 'slug', 'published', 'created_at', 'updated_at', 'views'],
    'comment': ['id', 'post_id', 'user_id', 'body', 'created_at', 'updated_at', 'parent_id'],
    'role': ['id', 'name', 'description', 'created_at'],
    'permission': ['id', 'name', 'description', 'created_at'],
    'category': ['id', 'name', 'slug', 'parent_id', 'created_at'],
    'tag': ['id', 'name', 'slug', 'created_at'],
    'followers': ['id', 'follower_id', 'followed_id', 'created_at'],
    'notification': ['id', 'user_id', 'type', 'message', 'read', 'created_at'],
    'message': ['id', 'sender_id', 'receiver_id', 'body', 'read', 'created_at'],
    'session': ['id', 'user_id', 'token', 'expires_at', 'created_at'],
    'api_key': ['id', 'user_id', 'key', 'name', 'created_at', 'last_used'],
    'upload': ['id', 'filename', 'path', 'user_id', 'created_at'],
    'task': ['id', 'name', 'status', 'result', 'created_at', 'completed_at'],
}


def generate_realistic_state(rng: np.random.Generator) -> List[float]:
    """Generate a realistic app state vector"""
    state = [0.0] * 170
    
    # Select random subset of tables (1-5)
    n_tables = rng.integers(1, 6)
    selected_tables = rng.choice(COMMON_TABLES, n_tables, replace=False).tolist()
    
    # Set table presence
    for i, table in enumerate(selected_tables[:5]):
        state[i] = 1.0
    
    # Set columns
    col_idx = 20
    for table in selected_tables:
        if table in COMMON_COLUMNS:
            cols = COMMON_COLUMNS[table]
            for col in cols[:10]:
                if col_idx < 170:
                    state[col_idx] = 1.0
                    col_idx += 1
    
    # Add some randomness
    noise = rng.normal(0, 0.05, 170)
    state = [s + n for s, n in zip(state, noise)]
    
    return state


def generate_realistic_edit(rng: np.random.Generator) -> tuple:
    """Generate a realistic edit and its outcome"""
    
    # Common edit types with realistic probabilities
    edit_types = [
        ('add_column', 0.30, 0.90),
        ('add_table', 0.15, 0.85),
        ('add_index', 0.15, 0.95),
        ('add_foreign_key', 0.10, 0.75),
        ('add_endpoint', 0.10, 0.85),
        ('add_test', 0.10, 0.95),
        ('modify_column', 0.05, 0.70),
        ('remove_column', 0.03, 0.60),
        ('rename_table', 0.02, 0.50),
    ]
    
    # Select edit type
    r = rng.random()
    cumulative = 0
    edit_type = 'add_column'
    success_rate = 0.90
    
    for et, prob, sr in edit_types:
        cumulative += prob
        if r <= cumulative:
            edit_type = et
            success_rate = sr
            break
    
    # Edit vector (32 dim)
    edit_vec = [0.0] * 32
    
    # One-hot encoding
    type_to_idx = {
        'add_column': 0, 'add_table': 1, 'add_index': 2,
        'add_foreign_key': 3, 'add_endpoint': 4, 'add_test': 5,
        'modify_column': 6, 'remove_column': 7, 'rename_table': 8,
    }
    edit_vec[type_to_idx.get(edit_type, 0)] = 1.0
    
    # Target hash (simple)
    target_idx = rng.integers(9, 32)
    edit_vec[target_idx] = 0.5
    
    # Determine outcome based on edit type's success rate
    outcome = 1.0 if rng.random() < success_rate else 0.0
    
    return edit_vec, outcome, edit_type


def generate_training_data(n_samples: int = 2000, seed: int = 42) -> List[Dict]:
    """Generate realistic training data"""
    
    rng = np.random.default_rng(seed)
    transitions = []
    
    for i in range(n_samples):
        # Generate state before
        state_before = generate_realistic_state(rng)
        
        # Generate edit
        edit, outcome, edit_type = generate_realistic_edit(rng)
        
        # Generate state after (apply edit)
        state_after = state_before.copy()
        
        # Simple state update
        if edit_type == 'add_column':
            for j in range(20, 50):
                if state_after[j] < 0.5:
                    state_after[j] = 1.0
                    break
        elif edit_type == 'add_table':
            for j in range(5):
                if state_after[j] < 0.5:
                    state_after[j] = 1.0
                    break
        
        # Add some noise to state_after
        noise = rng.normal(0, 0.02, 170)
        state_after = [s + n for s, n in zip(state_after, noise)]
        
        # Clamp values
        state_after = [max(0, min(1, s)) for s in state_after]
        
        transitions.append({
            'state_before': state_before,
            'edit': edit,
            'state_after': state_after,
            'outcome': outcome,
            'test_pass_rate': outcome,
            'edit_type': edit_type,
        })
    
    return transitions


def save_training_data(
    n_train: int = 2000,
    n_val: int = 250,
    n_test: int = 250,
    output_dir: str = "flask_app_data/collected/repos/flask"
):
    """Generate and save training data"""
    
    # Generate all data
    all_data = generate_training_data(n_train + n_val + n_test)
    
    # Shuffle
    rng = np.random.default_rng(42)
    rng.shuffle(all_data)
    
    # Split
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train+n_val]
    test_data = all_data[n_train+n_val:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    with open(output_path / "train.json", 'w') as f:
        json.dump(train_data, f)
    
    with open(output_path / "val.json", 'w') as f:
        json.dump(val_data, f)
    
    with open(output_path / "test.json", 'w') as f:
        json.dump(test_data, f)
    
    # Also save as transitions.json for compatibility
    with open(output_path / "transitions.json", 'w') as f:
        json.dump(all_data, f)
    
    print(f"Saved training data based on Flask patterns:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    print(f"  Output: {output_dir}")
    
    # Print outcome distribution
    n_outcomes = sum(1 for d in all_data if d['outcome'] > 0.5)
    print(f"  Outcome distribution: {n_outcomes}/{len(all_data)} positive ({100*n_outcomes/len(all_data):.1f}%)")
    
    # Print edit type distribution
    edit_types = {}
    for d in all_data:
        et = d.get('edit_type', 'unknown')
        edit_types[et] = edit_types.get(et, 0) + 1
    print(f"  Edit types: {edit_types}")


if __name__ == "__main__":
    save_training_data()

