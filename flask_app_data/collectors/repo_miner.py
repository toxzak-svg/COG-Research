"""
Repository Miner
Clones repositories and extracts schema changes, migrations, and test outcomes
"""

import os
import re
import json
import shutil
import subprocess
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from collections import defaultdict


@dataclass
class SchemaChange:
    """A database schema change extracted from commits"""
    commit_sha: str
    commit_message: str
    commit_date: str
    changed_files: List[str]
    
    # Schema changes
    tables_added: List[str] = field(default_factory=list)
    tables_removed: List[str] = field(default_factory=list)
    columns_added: Dict[str, List[str]] = field(default_factory=dict)  # table -> columns
    columns_removed: Dict[str, List[str]] = field(default_factory=dict)
    indices_added: List[str] = field(default_factory=list)
    
    # Outcome (will be filled by test outcome extractor)
    tests_passed: Optional[bool] = None
    test_count: int = 0
    coverage_delta: float = 0.0


@dataclass 
class AppSnapshot:
    """A snapshot of the app state at a specific point"""
    commit_sha: str
    commit_date: str
    
    # Schema state
    tables: List[str] = field(default_factory=list)
    columns: Dict[str, List[str]] = field(default_factory=dict)  # table -> columns
    indices: List[str] = field(default_factory=list)
    
    # Endpoint state
    endpoints: List[Dict] = field(default_factory=list)
    
    # Test state
    test_files: List[str] = field(default_factory=list)
    test_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'AppSnapshot':
        return cls(**d)


class RepoMiner:
    """
    Mine a repository for schema changes and app evolution
    
    Extracts:
    - Database schema (tables, columns, indices)
    - API endpoints
    - Test files
    - Commit history with diffs
    """
    
    def __init__(self, repos_dir: str = "flask_app_data/cloned_repos"):
        """
        Args:
            repos_dir: Directory to clone repos to
        """
        self.repos_dir = Path(repos_dir)
        self.repos_dir.mkdir(parents=True, exist_ok=True)
    
    def clone_repo(self, clone_url: str, repo_name: str) -> Optional[Path]:
        """
        Clone a repository
        
        Args:
            clone_url: Git clone URL
            repo_name: Name for the local directory
            
        Returns:
            Path to cloned repo, or None if failed
        """
        target_dir = self.repos_dir / repo_name.replace('/', '_')
        
        if target_dir.exists():
            print(f"  Repo already cloned: {repo_name}")
            return target_dir
        
        print(f"  Cloning {repo_name}...")
        
        try:
            # Shallow clone to save bandwidth
            result = subprocess.run(
                ["git", "clone", "--depth", "100", clone_url, str(target_dir)],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                print(f"  Failed to clone: {result.stderr}")
                return None
            
            return target_dir
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout cloning repo")
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def get_commit_history(self, repo_path: Path, max_commits: int = 200) -> List[Dict]:
        """
        Get commit history
        
        Args:
            repo_path: Path to cloned repo
            max_commits: Maximum number of commits to fetch
            
        Returns:
            List of commit dicts
        """
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={max_commits}", "--pretty=format:%H|%ad|%s", "--date=iso"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    parts = line.split('|', 2)
                    if len(parts) >= 3:
                        commits.append({
                            'sha': parts[0],
                            'date': parts[1],
                            'message': parts[2]
                        })
            
            return commits
            
        except Exception as e:
            print(f"  Error getting commit history: {e}")
            return []
    
    def get_commit_diff(self, repo_path: Path, commit_sha: str) -> Dict:
        """
        Get the diff for a specific commit
        
        Returns:
            Dict with 'files_changed', 'additions', 'deletions', 'patches'
        """
        try:
            # Get list of changed files
            result = subprocess.run(
                ["git", "show", "--name-status", "--pretty=format:", commit_sha],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {'files_changed': [], 'additions': 0, 'deletions': 0, 'patches': {}}
            
            files_changed = []
            additions = 0
            deletions = 0
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        status = parts[0]
                        filename = parts[1]
                        files_changed.append({'status': status, 'filename': filename})
                        
                        if status.startswith('A'):
                            additions += 1
                        elif status.startswith('D'):
                            deletions += 1
            
            return {
                'sha': commit_sha,
                'files_changed': files_changed,
                'additions': additions,
                'deletions': deletions
            }
            
        except Exception as e:
            print(f"  Error getting diff: {e}")
            return {'files_changed': [], 'additions': 0, 'deletions': 0, 'patches': {}}
    
    def extract_schema_from_models(self, repo_path: Path) -> AppSnapshot:
        """
        Extract current schema from models files
        
        Args:
            repo_path: Path to repo
            
        Returns:
            AppSnapshot with current state
        """
        snapshot = AppSnapshot(commit_sha="current", commit_date="")
        
        # Find model files
        model_files = list(repo_path.rglob("models.py"))
        model_files.extend(repo_path.rglob("model.py"))
        model_files.extend(list(repo_path.rglob("*models*.py")))
        
        for model_file in model_files:
            try:
                content = model_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract table names
                table_patterns = [
                    r'__tablename__\s*=\s*["\']([^"\']+)["\']',
                    r'table\s*\(\s*["\']([^"\']+)["\']',
                ]
                
                for pattern in table_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for table in matches:
                        if table not in snapshot.tables:
                            snapshot.tables.append(table)
                
                # Extract columns
                column_patterns = [
                    r'Column\s*\(\s*([^,\)]+)',
                    r'db\.Column\s*\(\s*([^,\)]+)',
                ]
                
                for table in snapshot.tables:
                    # Find class definition for this table
                    class_pattern = rf'class\s+(\w+).*?{table}'
                    class_match = re.search(class_pattern, content, re.DOTALL)
                    
                    if class_match:
                        class_body = class_match.group(0)
                        columns = re.findall(r'Column\s*\(\s*(\w+)', class_body)
                        snapshot.columns[table] = columns
                
                # Extract indices
                index_pattern = r'Index\s*\(\s*["\']([^"\']+)["\']'
                snapshot.indices.extend(re.findall(index_pattern, content, re.IGNORECASE))
                
            except Exception as e:
                continue
        
        # Find endpoints
        endpoint_files = list(repo_path.rglob("routes.py"))
        endpoint_files.extend(repo_path.rglob("api.py"))
        endpoint_files.extend(list(repo_path.rglob("views.py")))
        
        for endpoint_file in endpoint_files:
            try:
                content = endpoint_file.read_text(encoding='utf-8', errors='ignore')
                
                # Extract route decorators
                route_pattern = r'@.*?\.route\s*\(\s*["\']([^"\']+)["\']'
                routes = re.findall(route_pattern, content)
                
                for route in routes:
                    # Extract methods
                    methods_pattern = rf'@.*?\.route\([^)]+methods\s*=\s*\[([^\]]+)\]'
                    methods_match = re.search(methods_pattern, content)
                    
                    if methods_match:
                        methods = [m.strip().strip('"\'') for m in methods_match.group(1).split(',')]
                    else:
                        methods = ['GET']
                    
                    snapshot.endpoints.append({
                        'path': route,
                        'methods': methods
                    })
                
            except Exception as e:
                continue
        
        # Find test files
        test_dirs = list(repo_path.rglob("tests"))
        test_dirs.extend(repo_path.rglob("test"))
        
        for test_dir in test_dirs:
            if test_dir.is_dir():
                for test_file in test_dir.rglob("test_*.py"):
                    snapshot.test_files.append(str(test_file.relative_to(repo_path)))
                    snapshot.test_count += 1
        
        return snapshot
    
    def extract_schema_changes(self, repo_path: Path, commits: List[Dict]) -> List[SchemaChange]:
        """
        Extract schema changes from commit history
        
        Args:
            repo_path: Path to repo
            commits: List of commits
            
        Returns:
            List of SchemaChange objects
        """
        schema_changes = []
        
        for commit in commits:
            diff = self.get_commit_diff(repo_path, commit['sha'])
            
            # Filter for relevant files
            relevant_files = []
            for fc in diff['files_changed']:
                filename = fc['filename']
                if any(ext in filename for ext in ['models.py', 'model.py', 'migration', 'alembic']):
                    relevant_files.append(filename)
            
            if not relevant_files:
                continue
            
            # Create schema change record
            change = SchemaChange(
                commit_sha=commit['sha'],
                commit_message=commit['message'],
                commit_date=commit['date'],
                changed_files=relevant_files
            )
            
            # Analyze each changed file
            for filename in relevant_files:
                try:
                    # Get the diff for this file
                    result = subprocess.run(
                        ["git", "show", f"{commit['sha']}--", filename],
                        cwd=repo_path,
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode != 0:
                        continue
                    
                    patch = result.stdout
                    
                    # Extract table changes
                    if 'create table' in patch.lower():
                        tables = re.findall(r'create table [`"\']?(\w+)[`"\']?', patch, re.IGNORECASE)
                        change.tables_added.extend(tables)
                    
                    if 'drop table' in patch.lower():
                        tables = re.findall(r'drop table [`"\']?(\w+)[`"\']?', patch, re.IGNORECASE)
                        change.tables_removed.extend(tables)
                    
                    # Extract column changes
                    add_col_pattern = r'alter\s+table\s+(\w+)\s+add\s+column\s+(\w+)'
                    for match in re.finditer(add_col_pattern, patch, re.IGNORECASE):
                        table, column = match.groups()
                        if table not in change.columns_added:
                            change.columns_added[table] = []
                        change.columns_added[table].append(column)
                    
                    drop_col_pattern = r'alter\s+table\s+(\w+)\s+drop\s+column\s+(\w+)'
                    for match in re.finditer(drop_col_pattern, patch, re.IGNORECASE):
                        table, column = match.groups()
                        if table not in change.columns_removed:
                            change.columns_removed[table] = []
                        change.columns_removed[table].append(column)
                    
                except Exception as e:
                    continue
            
            # Only keep changes that actually have schema modifications
            if (change.tables_added or change.tables_removed or 
                change.columns_added or change.columns_removed or
                change.indices_added):
                schema_changes.append(change)
        
        return schema_changes
    
    def mine_repo(self, repo_info: Dict) -> Dict:
        """
        Mine a repository for schema changes
        
        Args:
            repo_info: Dict with repo info from GitHubCollector
            
        Returns:
            Dict with 'snapshots', 'schema_changes', 'commits'
        """
        repo_name = repo_info['full_name']
        clone_url = repo_info['clone_url']
        
        print(f"\nMining {repo_name}...")
        
        # Clone repo
        repo_path = self.clone_repo(clone_url, repo_name)
        if not repo_path:
            return {'error': 'Failed to clone'}
        
        # Get commit history
        commits = self.get_commit_history(repo_path)
        print(f"  Found {len(commits)} commits")
        
        # Extract current state
        current_snapshot = self.extract_schema_from_models(repo_path)
        
        # Extract schema changes
        schema_changes = self.extract_schema_changes(repo_path, commits)
        print(f"  Found {len(schema_changes)} schema changes")
        
        # Try to get test outcomes from CI
        # (This would require running workflows or parsing logs)
        
        return {
            'repo_name': repo_name,
            'current_snapshot': current_snapshot.to_dict(),
            'schema_changes': [asdict(sc) for sc in schema_changes],
            'total_commits': len(commits),
            'cloned_path': str(repo_path)
        }


def main():
    """Demo: Mine a sample repository"""
    miner = RepoMiner()
    
    # This would normally come from GitHubCollector
    sample_repo = {
        'full_name': 'example/flask-app',
        'clone_url': 'https://github.com/example/flask-app.git'
    }
    
    # Mine the repo
    result = miner.mine_repo(sample_repo)
    
    print(f"\nMining result:")
    print(f"  Repo: {result.get('repo_name', 'N/A')}")
    print(f"  Schema changes: {len(result.get('schema_changes', []))}")
    print(f"  Commits: {result.get('total_commits', 0)}")


if __name__ == "__main__":
    main()

