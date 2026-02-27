"""
App State Representation for Demo

Represents the current state of a Flask application.
"""


class AppState:
    """Represents the state of a Flask application."""
    
    def __init__(self):
        self.schema = {
            'tables': [],
            'columns': {},
            'indices': {},
        }
        self.endpoints = []
        self.tests = {
            'total': 0,
            'passing': 0,
            'coverage': 0.0,
        }
        self.auth_config = {
            'enabled': False,
            'protected_endpoints': [],
        }
    
    def initialize_simple_flask_app(self):
        """Set up a simple Flask app with user authentication."""
        # Database schema
        self.schema['tables'] = ['User']
        self.schema['columns'] = {
            'User': ['id', 'username', 'password_hash']
        }
        self.schema['indices'] = {
            'User': ['id']
        }
        
        # API endpoints
        self.endpoints = [
            {'path': '/api/register', 'methods': ['POST'], 'auth': False},
            {'path': '/api/login', 'methods': ['POST'], 'auth': False},
        ]
        
        # Tests
        self.tests = {
            'total': 5,
            'passing': 5,
            'coverage': 0.82,
        }
        
        # Auth config
        self.auth_config = {
            'enabled': True,
            'protected_endpoints': [],
        }
    
    def to_dict(self):
        """Convert to dictionary for encoding."""
        return {
            'schema': self.schema,
            'endpoints': self.endpoints,
            'tests': self.tests,
            'auth_config': self.auth_config,
        }
