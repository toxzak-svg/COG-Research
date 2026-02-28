"""
GitHub Repository Collector
Searches GitHub for Flask repositories with database migrations and CI/CD
"""

import os
import json
import time
import requests
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class RepoInfo:
    """Information about a collected repository"""
    name: str
    full_name: str
    owner: str
    description: Optional[str]
    stars: int
    forks: int
    language: str
    created_at: str
    updated_at: str
    has_workflows: bool
    has_migrations: bool
    has_models: bool
    has_tests: bool
    clone_url: str
    default_branch: str
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'RepoInfo':
        return cls(**d)


class GitHubCollector:
    """
    Collect Flask repositories from GitHub with relevant features
    
    Filters for:
    - Flask web apps
    - Database migrations (Alembic/AutoMigrate)
    - CI/CD workflows
    - Test files
    """
    
    def __init__(self, token: Optional[str] = None, data_dir: str = "flask_app_data/repos"):
        """
        Args:
            token: GitHub personal access token (optional, for higher rate limits)
            data_dir: Directory to store collected repo info
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Flask-App-Collector/1.0"
        })
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        
        # Cache for rate limiting
        self.rate_limit_remaining = None
        self.rate_limit_reset = None
    
    def _check_rate_limit(self):
        """Check and handle rate limiting"""
        if self.rate_limit_remaining is not None and self.rate_limit_remaining < 5:
            wait_time = self.rate_limit_reset - time.time() if self.rate_limit_reset else 60
            if wait_time > 0:
                print(f"Rate limit low, waiting {wait_time:.0f}s...")
                time.sleep(min(wait_time, 60))
        
        # Check current rate limit
        resp = self.session.get("https://api.github.com/rate_limit")
        if resp.status_code == 200:
            limits = resp.json()
            self.rate_limit_remaining = limits['resources']['search']['remaining']
            self.rate_limit_reset = limits['resources']['search']['reset']
    
    def _search_repos(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search GitHub repositories using query language
        
        Args:
            query: GitHub search query
            max_results: Maximum number of results
            
        Returns:
            List of repository dicts
        """
        self._check_rate_limit()
        
        repos = []
        page = 1
        per_page = 100
        
        while len(repos) < max_results:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            
            resp = self.session.get(
                "https://api.github.com/search/repositories",
                params=params
            )
            
            if resp.status_code != 200:
                print(f"Error: {resp.status_code} - {resp.text}")
                break
            
            data = resp.json()
            items = data.get("items", [])
            
            if not items:
                break
            
            repos.extend(items)
            
            # Check rate limit from response headers
            if 'X-RateLimit-Remaining' in resp.headers:
                self.rate_limit_remaining = int(resp.headers['X-RateLimit-Remaining'])
                self.rate_limit_reset = int(resp.headers['X-RateLimit-Reset'])
            
            # Check if we've reached the end
            if len(items) < per_page:
                break
            
            page += 1
            time.sleep(1)  # Be nice to the API
        
        return repos[:max_results]
    
    def search_flask_repos(self, min_stars: int = 10, max_repos: int = 200) -> List[RepoInfo]:
        """
        Search for Flask repositories with database migrations
        
        Args:
            min_stars: Minimum number of stars
            max_repos: Maximum number of repos to collect
            
        Returns:
            List of RepoInfo objects
        """
        print(f"Searching for Flask repos with stars >= {min_stars}...")
        
        # Build search query
        # Looking for: Flask + either migrations OR models + tests
        queries = [
            # Flask with migrations
            f"flask in:name description language:Python stars:>{min_stars} pushed:>2023-01-01",
        ]
        
        all_repos = []
        
        for query in queries:
            print(f"  Query: {query[:80]}...")
            repos = self._search_repos(query, max_repos)
            all_repos.extend(repos)
            
            # Deduplicate
            seen = set()
            unique_repos = []
            for repo in all_repos:
                if repo['full_name'] not in seen:
                    seen.add(repo['full_name'])
                    unique_repos.append(repo)
            all_repos = unique_repos
            
            if len(all_repos) >= max_repos:
                break
        
        print(f"Found {len(all_repos)} potential repos")
        
        # Now check each repo for required features
        print("Checking repos for migrations, models, and tests...")
        
        repo_infos = []
        for i, repo in enumerate(all_repos[:max_repos]):
            if (i + 1) % 20 == 0:
                print(f"  Checked {i+1}/{len(all_repos)} repos...")
            
            features = self._check_repo_features(repo['full_name'])
            
            repo_info = RepoInfo(
                name=repo['name'],
                full_name=repo['full_name'],
                owner=repo['owner']['login'],
                description=repo.get('description'),
                stars=repo['stargazers_count'],
                forks=repo['forks_count'],
                language=repo.get('language'),
                created_at=repo['created_at'],
                updated_at=repo['updated_at'],
                has_workflows=features['has_workflows'],
                has_migrations=features['has_migrations'],
                has_models=features['has_models'],
                has_tests=features['has_tests'],
                clone_url=repo['clone_url'],
                default_branch=repo.get('default_branch', 'main')
            )
            
            repo_infos.append(repo_info)
            
            # Save progress
            self._save_repo_info(repo_info)
        
        # Filter for repos that have the features we need
        qualified = [r for r in repo_infos if r.has_migrations or (r.has_models and r.has_tests)]
        
        print(f"\nQualified repos (with migrations or models+tests): {len(qualified)}/{len(repo_infos)}")
        
        return qualified
    
    def _check_repo_features(self, full_name: str) -> Dict[str, bool]:
        """
        Check if a repo has the features we need
        
        Returns:
            Dict with has_workflows, has_migrations, has_models, has_tests
        """
        features = {
            'has_workflows': False,
            'has_migrations': False,
            'has_models': False,
            'has_tests': False
        }
        
        # Get repository contents
        resp = self.session.get(f"https://api.github.com/repos/{full_name}/contents")
        
        if resp.status_code != 200:
            return features
        
        try:
            contents = resp.json()
        except:
            return features
        
        if not isinstance(contents, list):
            return features
        
        # Check for key files/directories
        names = [c.get('name', '').lower() for c in contents]
        
        # Check for workflows
        if '.github' in names:
            features['has_workflows'] = True
        
        # Check for migrations
        migration_indicators = ['migrations', 'alembic', 'migrate']
        features['has_migrations'] = any(ind in ' '.join(names) for ind in migration_indicators)
        
        # Check for models
        model_indicators = ['models', 'model.py', 'models.py']
        features['has_models'] = any(ind in ' '.join(names) for ind in model_indicators)
        
        # Check for tests
        test_indicators = ['tests', 'test_', '_test.py', 'pytest.ini', 'tox.ini']
        features['has_tests'] = any(ind in ' '.join(names) for ind in test_indicators)
        
        return features
    
    def _save_repo_info(self, repo_info: RepoInfo):
        """Save repo info to JSON file"""
        output_file = self.data_dir / f"{repo_info.full_name.replace('/', '_')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(repo_info.to_dict(), f, indent=2)
    
    def load_collected_repos(self) -> List[RepoInfo]:
        """Load all previously collected repos"""
        repos = []
        
        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                repos.append(RepoInfo.from_dict(data))
        
        return repos
    
    def get_filtered_repos(
        self,
        min_stars: int = 10,
        require_migrations: bool = True,
        require_tests: bool = True
    ) -> List[RepoInfo]:
        """
        Get repos that match criteria
        
        Args:
            min_stars: Minimum stars
            require_migrations: Must have migrations
            require_tests: Must have tests
            
        Returns:
            Filtered list of RepoInfo
        """
        repos = self.load_collected_repos()
        
        filtered = []
        for repo in repos:
            if repo.stars < min_stars:
                continue
            if require_migrations and not repo.has_migrations:
                continue
            if require_tests and not repo.has_tests:
                continue
            filtered.append(repo)
        
        print(f"Filtered repos: {len(filtered)}/{len(repos)} (stars>={min_stars}, migrations={require_migrations}, tests={require_tests})")
        
        return filtered


def main():
    """Demo: Search for Flask repos"""
    collector = GitHubCollector()
    
    # Search for repos
    repos = collector.search_flask_repos(min_stars=10, max_repos=100)
    
    # Show results
    print(f"\nFound {len(repos)} qualified repos:")
    
    for repo in repos[:10]:
        features = []
        if repo.has_migrations:
            features.append("migrations")
        if repo.has_models:
            features.append("models")
        if repo.has_tests:
            features.append("tests")
        if repo.has_workflows:
            features.append("CI/CD")
        
        print(f"  - {repo.full_name}: {repo.stars} stars, {', '.join(features)}")
    
    # Save summary
    output_file = "flask_app_data/collected_repos.json"
    with open(output_file, 'w') as f:
        json.dump([r.to_dict() for r in repos], f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()

