"""
Data Schemas for Flask App Evolution

Defines the data structures for:
- AppState: Represents the state of a Flask app
- EditRecord: Represents a change to the app
- Transition: (state, edit, outcome) tuple
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class EditType(Enum):
    """Types of edits that can be made to an app"""
    # Schema edits
    ADD_TABLE = "add_table"
    DROP_TABLE = "drop_table"
    RENAME_TABLE = "rename_table"
    ADD_COLUMN = "add_column"
    DROP_COLUMN = "drop_column"
    RENAME_COLUMN = "rename_column"
    ADD_INDEX = "add_index"
    DROP_INDEX = "drop_index"
    ADD_FOREIGN_KEY = "add_foreign_key"
    DROP_FOREIGN_KEY = "drop_foreign_key"
    
    # Endpoint edits
    ADD_ENDPOINT = "add_endpoint"
    REMOVE_ENDPOINT = "remove_endpoint"
    MODIFY_ENDPOINT = "modify_endpoint"
    ADD_AUTH = "add_auth"
    REMOVE_AUTH = "remove_auth"
    
    # Test edits
    ADD_TEST = "add_test"
    REMOVE_TEST = "remove_test"
    MODIFY_TEST = "modify_test"
    
    # Configuration edits
    ADD_DEPENDENCY = "add_dependency"
    REMOVE_DEPENDENCY = "remove_dependency"
    UPDATE_CONFIG = "update_config"


@dataclass
class SchemaState:
    """Database schema state"""
    tables: List[str] = field(default_factory=list)
    columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> columns
    indices: List[str] = field(default_factory=list)
    foreign_keys: List[Dict] = field(default_factory=list)
    
    def to_vector(self, max_tables: int = 20, max_columns_per_table: int = 20) -> List[float]:
        """
        Convert to fixed-size vector
        
        Output: (max_tables + max_tables * max_columns_per_table + max_indices) dims
        """
        vec = []
        
        # Table presence (max_tables)
        for i in range(max_tables):
            vec.append(1.0 if i < len(self.tables) else 0.0)
        
        # Column presence (max_tables * max_columns_per_table)
        for i in range(max_tables):
            if i < len(self.tables):
                table_name = self.tables[i]
                table_cols = self.columns.get(table_name, [])
                for j in range(max_columns_per_table):
                    vec.append(1.0 if j < len(table_cols) else 0.0)
            else:
                vec.extend([0.0] * max_columns_per_table)
        
        # Index presence (max_indices = max_tables * 2)
        for i in range(max_tables * 2):
            vec.append(1.0 if i < len(self.indices) else 0.0)
        
        return vec
    
    @classmethod
    def from_sqlalchemy_models(cls, content: str) -> 'SchemaState':
        """Parse SQLAlchemy model definitions"""
        import re
        
        schema = cls()
        
        # Find table names
        table_pattern = r'__tablename__\s*=\s*["\']([^"\']+)["\']'
        schema.tables = re.findall(table_pattern, content)
        
        # Find columns for each table
        for table in schema.tables:
            class_pattern = rf'class\s+(\w+).*?{table}.*?(?=class\s|\Z)'
            class_match = re.search(class_pattern, content, re.DOTALL)
            
            if class_match:
                class_body = class_match.group(0)
                col_pattern = r'(\w+)\s*=\s*Column'
                columns = re.findall(col_pattern, class_body)
                # Filter out common non-column definitions
                columns = [c for c in columns if c not in ['id', 'created_at', 'updated_at'] or True]
                schema.columns[table] = columns
        
        # Find indices
        index_pattern = r'Index\s*\(\s*["\']([^"\']+)["\']'
        schema.indices = re.findall(index_pattern, content, re.IGNORECASE)
        
        return schema


@dataclass
class EndpointState:
    """API endpoint state"""
    endpoints: List[Dict] = field(default_factory=list)  # {path, methods, auth}
    
    def to_vector(self, max_endpoints: int = 50) -> List[float]:
        """
        Convert to fixed-size vector
        
        Output: max_endpoints * 4 dims (path_hash, get, post, auth)
        """
        import hashlib
        
        vec = []
        
        for i in range(max_endpoints):
            if i < len(self.endpoints):
                ep = self.endpoints[i]
                
                # Hash of path (to keep it fixed-size)
                path_hash = int(hashlib.md5(ep['path'].encode()).hexdigest()[:8], 16)
                path_hash = path_hash / (16**8)  # Normalize to [0, 1]
                vec.append(path_hash)
                
                # Method flags
                vec.append(1.0 if 'GET' in ep.get('methods', []) else 0.0)
                vec.append(1.0 if 'POST' in ep.get('methods', []) else 0.0)
                
                # Auth flag
                vec.append(1.0 if ep.get('auth', False) else 0.0)
            else:
                vec.extend([0.0] * 4)
        
        return vec
    
    @classmethod
    def from_flask_routes(cls, content: str) -> 'EndpointState':
        """Parse Flask route definitions"""
        import re
        
        state = cls()
        
        # Find route decorators
        route_pattern = r'@(?:app|blueprint)\.route\s*\(\s*["\']([^"\']+)["\'][^)]*\)'
        routes = re.findall(route_pattern, content)
        
        for route in routes:
            # Find methods
            methods_match = re.search(
                rf'@.*?route\([^)]+methods\s*=\s*\[([^\]]+)\]',
                content
            )
            
            if methods_match:
                methods = [m.strip().strip('"\'') for m in methods_match.group(1).split(',')]
            else:
                methods = ['GET']
            
            # Check for auth
            auth = 'login_required' in content or 'auth_required' in content
            
            state.endpoints.append({
                'path': route,
                'methods': methods,
                'auth': auth
            })
        
        return state


@dataclass
class TestState:
    """Test suite state"""
    test_files: List[str] = field(default_factory=list)
    test_count: int = 0
    passing: bool = True
    
    def to_vector(self, max_test_files: int = 20) -> List[float]:
        """
        Convert to fixed-size vector
        
        Output: max_test_files + 3 dims
        """
        vec = []
        
        # Test file presence
        for i in range(max_test_files):
            vec.append(1.0 if i < len(self.test_files) else 0.0)
        
        # Test count (normalized)
        vec.append(min(self.test_count / 100.0, 1.0))
        
        # Pass rate (normalized)
        vec.append(1.0 if self.passing else 0.0)
        
        # Coverage (placeholder)
        vec.append(0.8)  # Default coverage
        
        return vec


@dataclass
class AppState:
    """
    Complete state of a Flask application
    
    This is the main data structure representing what the app looks like
    at a specific point in its evolution.
    """
    repo_name: str = ""
    commit_sha: str = ""
    commit_date: str = ""
    
    schema: SchemaState = field(default_factory=SchemaState)
    endpoints: EndpointState = field(default_factory=EndpointState)
    tests: TestState = field(default_factory=TestState)
    
    # Additional metadata
    dependencies: List[str] = field(default_factory=list)
    python_version: str = ""
    flask_version: str = ""
    
    def to_vector(self, state_dim: int = 170) -> List[float]:
        """
        Convert complete state to fixed-size vector
        
        Total dims:
        - Schema: 20 + 20*20 + 40 = 460 (but we'll compress)
        - Endpoints: 50*4 = 200
        - Tests: 20 + 3 = 23
        - Total input: ~683 dims → compress to state_dim with VAE
        """
        schema_vec = self.schema.to_vector()
        endpoint_vec = self.endpoints.to_vector()
        test_vec = self.tests.to_vector()
        
        # Concatenate and pad/truncate to state_dim
        combined = schema_vec + endpoint_vec + test_vec
        
        if len(combined) > state_dim:
            return combined[:state_dim]
        else:
            return combined + [0.0] * (state_dim - len(combined))
    
    def to_dict(self) -> Dict:
        return {
            'repo_name': self.repo_name,
            'commit_sha': self.commit_sha,
            'commit_date': self.commit_date,
            'schema': {
                'tables': self.schema.tables,
                'columns': self.schema.columns,
                'indices': self.schema.indices,
            },
            'endpoints': self.endpoints.endpoints,
            'tests': {
                'test_files': self.tests.test_files,
                'test_count': self.tests.test_count,
                'passing': self.tests.passing,
            },
            'dependencies': self.dependencies,
            'python_version': self.python_version,
            'flask_version': self.flask_version,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AppState':
        state = cls()
        state.repo_name = d.get('repo_name', '')
        state.commit_sha = d.get('commit_sha', '')
        state.commit_date = d.get('commit_date', '')
        
        schema_data = d.get('schema', {})
        state.schema = SchemaState(
            tables=schema_data.get('tables', []),
            columns=schema_data.get('columns', {}),
            indices=schema_data.get('indices', []),
        )
        
        state.endpoints = EndpointState(endpoints=d.get('endpoints', []))
        
        test_data = d.get('tests', {})
        state.tests = TestState(
            test_files=test_data.get('test_files', []),
            test_count=test_data.get('test_count', 0),
            passing=test_data.get('passing', True),
        )
        
        state.dependencies = d.get('dependencies', [])
        state.python_version = d.get('python_version', '')
        state.flask_version = d.get('flask_version', '')
        
        return state
    
    def __len__(self) -> int:
        """Return vector dimension"""
        return 170  # Default state dimension


@dataclass
class EditRecord:
    """
    Record of an edit made to the app
    
    This represents what was changed between two states.
    """
    edit_id: str = ""
    edit_type: EditType = EditType.ADD_COLUMN
    
    # Edit details
    target: str = ""  # table name, endpoint path, etc.
    field: str = ""  # column name, method, etc.
    value: Any = None  # new value
    
    # Context
    commit_sha: str = ""
    commit_message: str = ""
    repo_name: str = ""
    
    def to_vector(self, edit_dim: int = 32) -> List[float]:
        """
        Convert to fixed-size vector
        
        Output: edit_dim dims
        """
        import hashlib
        
        # One-hot encode edit type
        type_vec = [0.0] * len(EditType)
        try:
            type_vec[self.edit_type.value] = 1.0
        except:
            pass
        
        # Hash of target + field
        target_str = f"{self.target}:{self.field}"
        target_hash = int(hashlib.md5(target_str.encode()).hexdigest()[:8], 16)
        target_hash = target_hash / (16**8)
        
        # Value hash
        value_str = str(self.value) if self.value else ""
        value_hash = int(hashlib.md5(value_str.encode()).hexdigest()[:8], 16)
        value_hash = value_hash / (16**8)
        
        # Combine
        combined = type_vec + [target_hash, value_hash]
        
        if len(combined) > edit_dim:
            return combined[:edit_dim]
        else:
            return combined + [0.0] * (edit_dim - len(combined))
    
    def to_dict(self) -> Dict:
        return {
            'edit_id': self.edit_id,
            'edit_type': self.edit_type.value if isinstance(self.edit_type, EditType) else str(self.edit_type),
            'target': self.target,
            'field': self.field,
            'value': str(self.value) if self.value else None,
            'commit_sha': self.commit_sha,
            'commit_message': self.commit_message,
            'repo_name': self.repo_name,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'EditRecord':
        record = cls()
        record.edit_id = d.get('edit_id', '')
        
        edit_type = d.get('edit_type', '')
        if isinstance(edit_type, str):
            try:
                record.edit_type = EditType(edit_type)
            except:
                record.edit_type = EditType.ADD_COLUMN
        
        record.target = d.get('target', '')
        record.field = d.get('field', '')
        record.value = d.get('value')
        record.commit_sha = d.get('commit_sha', '')
        record.commit_message = d.get('commit_message', '')
        record.repo_name = d.get('repo_name', '')
        
        return record


@dataclass 
class Transition:
    """
    A state transition: (state_t, edit, state_t+1, outcome)
    
    This is the main training data format for the world-model.
    """
    state_before: AppState = field(default_factory=AppState)
    edit: EditRecord = field(default_factory=EditRecord)
    state_after: AppState = field(default_factory=AppState)
    
    # Outcome metrics
    tests_passed: bool = True
    test_pass_rate: float = 1.0
    test_count: int = 0
    
    # Optional: coverage delta
    coverage_delta: float = 0.0
    
    def to_training_tuple(self, state_dim: int = 170, edit_dim: int = 32) -> Dict:
        """
        Convert to training tuple for world-model
        
        Returns:
            {
                'state_before': vector,
                'edit': vector,
                'state_after': vector,
                'outcome': scalar (1.0 = success, 0.0 = failure)
            }
        """
        return {
            'state_before': self.state_before.to_vector(state_dim),
            'edit': self.edit.to_vector(edit_dim),
            'state_after': self.state_after.to_vector(state_dim),
            'outcome': 1.0 if self.tests_passed else 0.0,
            'test_pass_rate': self.test_pass_rate,
            'test_count': self.test_count,
        }
    
    def to_dict(self) -> Dict:
        return {
            'state_before': self.state_before.to_dict(),
            'edit': self.edit.to_dict(),
            'state_after': self.state_after.to_dict(),
            'tests_passed': self.tests_passed,
            'test_pass_rate': self.test_pass_rate,
            'test_count': self.test_count,
            'coverage_delta': self.coverage_delta,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Transition':
        transition = cls()
        transition.state_before = AppState.from_dict(d.get('state_before', {}))
        transition.edit = EditRecord.from_dict(d.get('edit', {}))
        transition.state_after = AppState.from_dict(d.get('state_after', {}))
        transition.tests_passed = d.get('tests_passed', True)
        transition.test_pass_rate = d.get('test_pass_rate', 1.0)
        transition.test_count = d.get('test_count', 0)
        transition.coverage_delta = d.get('coverage_delta', 0.0)
        
        return transition


# Helper functions
def serialize_state(state: AppState) -> str:
    """Serialize state to JSON string"""
    return json.dumps(state.to_dict())


def deserialize_state(json_str: str) -> AppState:
    """Deserialize state from JSON string"""
    return AppState.from_dict(json.loads(json_str))


def serialize_transition(transition: Transition) -> str:
    """Serialize transition to JSON string"""
    return json.dumps(transition.to_dict())


def deserialize_transition(json_str: str) -> Transition:
    """Deserialize transition from JSON string"""
    return Transition.from_dict(json.loads(json_str))

