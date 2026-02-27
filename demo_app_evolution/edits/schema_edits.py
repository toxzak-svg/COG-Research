"""
Schema Edit Operations

Database schema changes (add column, remove column, add index, etc.)
"""

from demo_app_evolution.edits.base import Edit


class AddColumnEdit(Edit):
    """Add a column to a database table."""
    
    def __init__(self, table, column, type, nullable=True):
        super().__init__()
        self.edit_type = 'add_column'
        self.table = table
        self.column = column
        self.type = type
        self.nullable = nullable
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        if self.table in app_state.schema['columns']:
            app_state.schema['columns'][self.table].append(self.column)
        
        # Update test outcomes (simplified simulation)
        # In reality, this would run actual tests
        app_state.tests['total'] += 1
        app_state.tests['passing'] += 1  # Assume tests pass for this simple edit
        app_state.tests['coverage'] = (app_state.tests['coverage'] * (app_state.tests['total'] - 1) + 0.9) / app_state.tests['total']
    
    def short_description(self):
        return f"Add column {self.table}.{self.column}"


class RemoveColumnEdit(Edit):
    """Remove a column from a database table."""
    
    def __init__(self, table, column):
        super().__init__()
        self.edit_type = 'remove_column'
        self.table = table
        self.column = column
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        if self.table in app_state.schema['columns']:
            if self.column in app_state.schema['columns'][self.table]:
                app_state.schema['columns'][self.table].remove(self.column)
        
        # This is a risky edit - might break tests
        app_state.tests['total'] += 1
        app_state.tests['passing'] += 0  # Assume it breaks a test
        app_state.tests['coverage'] *= 0.95  # Coverage drops
    
    def short_description(self):
        return f"Remove column {self.table}.{self.column}"


class AddIndexEdit(Edit):
    """Add an index to a table."""
    
    def __init__(self, table, column):
        super().__init__()
        self.edit_type = 'add_index'
        self.table = table
        self.column = column
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        if self.table not in app_state.schema['indices']:
            app_state.schema['indices'][self.table] = []
        
        if self.column not in app_state.schema['indices'][self.table]:
            app_state.schema['indices'][self.table].append(self.column)
        
        # Indices usually don't require new tests
        pass
    
    def short_description(self):
        return f"Add index on {self.table}.{self.column}"
