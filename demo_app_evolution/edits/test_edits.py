"""
Test Edit Operations

Test suite changes (add test, remove test, update test, etc.)
"""

from demo_app_evolution.edits.base import Edit


class AddTestEdit(Edit):
    """Add a new test."""
    
    def __init__(self, test_name, file='test_app.py'):
        super().__init__()
        self.edit_type = 'add_test'
        self.test_name = test_name
        self.file = file
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        app_state.tests['total'] += 1
        app_state.tests['passing'] += 1  # Assume new test passes
        
        # Adding tests increases coverage
        app_state.tests['coverage'] = min(
            app_state.tests['coverage'] + 0.02,
            0.95
        )
    
    def short_description(self):
        return f"Add test {self.test_name}"


class RemoveTestEdit(Edit):
    """Remove a test."""
    
    def __init__(self, test_name):
        super().__init__()
        self.edit_type = 'remove_test'
        self.test_name = test_name
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        if app_state.tests['total'] > 0:
            app_state.tests['total'] -= 1
            # Assume the removed test was passing
            if app_state.tests['passing'] > 0:
                app_state.tests['passing'] -= 1
        
        # Removing tests decreases coverage
        app_state.tests['coverage'] = max(
            app_state.tests['coverage'] - 0.03,
            0.0
        )
    
    def short_description(self):
        return f"Remove test {self.test_name}"


class UpdateTestEdit(Edit):
    """Update an existing test."""
    
    def __init__(self, test_name, change_description):
        super().__init__()
        self.edit_type = 'update_test'
        self.test_name = test_name
        self.change_description = change_description
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        # Test updates might temporarily break tests
        # Assume 80% chance it still passes
        import random
        if random.random() > 0.2:
            pass  # Still passing
        else:
            app_state.tests['passing'] = max(0, app_state.tests['passing'] - 1)
    
    def short_description(self):
        return f"Update test {self.test_name}"
