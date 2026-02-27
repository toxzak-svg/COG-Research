"""
Base Edit Class

All edit types inherit from this base class.
"""


class Edit:
    """Base class for all edit operations."""
    
    def __init__(self):
        self.edit_type = 'unknown'
    
    def apply(self, app_state):
        """Apply this edit to the app state."""
        raise NotImplementedError
    
    def short_description(self):
        """Return a short description of this edit."""
        raise NotImplementedError
    
    def to_vector(self):
        """Convert edit to vector representation for model input."""
        raise NotImplementedError
