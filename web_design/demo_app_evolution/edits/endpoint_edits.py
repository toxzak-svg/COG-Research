"""
Endpoint Edit Operations

API endpoint changes (add endpoint, remove endpoint, modify auth, etc.)
"""

from demo_app_evolution.edits.base import Edit


class AddEndpointEdit(Edit):
    """Add a new API endpoint."""
    
    def __init__(self, path, method, auth=False, metadata=None):
        super().__init__()
        self.edit_type = 'add_endpoint'
        self.path = path
        self.method = method
        self.auth = auth
        self.metadata = metadata or {}
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        new_endpoint = {
            'path': self.path,
            'methods': [self.method],
            'auth': self.auth,
        }
        app_state.endpoints.append(new_endpoint)
        
        if self.auth:
            app_state.auth_config['protected_endpoints'].append(self.path)
        
        # Update tests
        app_state.tests['total'] += 2  # Typically add 2 tests per endpoint
        app_state.tests['passing'] += 2
        app_state.tests['coverage'] = (app_state.tests['coverage'] * (app_state.tests['total'] - 2) + 1.8) / app_state.tests['total']
    
    def short_description(self):
        auth_str = " (auth)" if self.auth else ""
        return f"Add endpoint {self.method} {self.path}{auth_str}"


class RemoveEndpointEdit(Edit):
    """Remove an API endpoint."""
    
    def __init__(self, path):
        super().__init__()
        self.edit_type = 'remove_endpoint'
        self.path = path
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        app_state.endpoints = [ep for ep in app_state.endpoints if ep['path'] != self.path]
        
        if self.path in app_state.auth_config['protected_endpoints']:
            app_state.auth_config['protected_endpoints'].remove(self.path)
        
        # Removing endpoints might break tests
        app_state.tests['total'] += 1
        app_state.tests['passing'] += 0  # Might break integration tests
    
    def short_description(self):
        return f"Remove endpoint {self.path}"


class ModifyEndpointAuthEdit(Edit):
    """Modify authentication requirement for an endpoint."""
    
    def __init__(self, path, require_auth):
        super().__init__()
        self.edit_type = 'modify_endpoint_auth'
        self.path = path
        self.require_auth = require_auth
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        for endpoint in app_state.endpoints:
            if endpoint['path'] == self.path:
                endpoint['auth'] = self.require_auth
                
                if self.require_auth:
                    if self.path not in app_state.auth_config['protected_endpoints']:
                        app_state.auth_config['protected_endpoints'].append(self.path)
                else:
                    if self.path in app_state.auth_config['protected_endpoints']:
                        app_state.auth_config['protected_endpoints'].remove(self.path)
        
        # Auth changes need security tests
        app_state.tests['total'] += 1
        app_state.tests['passing'] += 1
    
    def short_description(self):
        return f"Set auth={self.require_auth} for {self.path}"
