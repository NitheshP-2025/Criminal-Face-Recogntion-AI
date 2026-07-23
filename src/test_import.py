# test_import.py
"""Test if imports work"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing imports...")

try:
    from src import CriminalFaceDataset
    print("✅ CriminalFaceDataset imported successfully")
except ImportError as e:
    print(f"❌ Failed to import CriminalFaceDataset: {e}")

try:
    from src import FeatureScaler
    print("✅ FeatureScaler imported successfully")
except ImportError as e:
    print(f"❌ Failed to import FeatureScaler: {e}")

print("\nChecking files...")
print(f"  src/__init__.py exists: {os.path.exists('src/__init__.py')}")
print(f"  src/data_loader.py exists: {os.path.exists('src/data_loader.py')}")

print("\n✅ Test complete!")
