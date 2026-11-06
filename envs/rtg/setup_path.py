"""Add rtg/ directory to sys.path so BBM's bare imports work."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
