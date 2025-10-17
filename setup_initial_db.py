# setup_initial_db.py
# -*- coding: utf-8 -*-
"""
Initial setup script to load all Wikipedia data into vector database.
Run this ONCE on first setup, then it won't need to reload.
"""
import sys
from pathlib import Path

# Change to match walle_rag.py structure
sys.path.insert(0, str(Path(__file__).parent))

from walle_rag import load_json_files, VECTOR_DB_DIR, DATA_DIR

print("=" * 60)
print("WALL-E Initial Database Setup")
print("=" * 60)
print("\nThis will load all Wikipedia data into the vector database.")
print("This only needs to be done ONCE and may take 10-30 minutes.")
print(f"\nData directory: {DATA_DIR}")
print(f"Vector DB directory: {VECTOR_DB_DIR}")

# Count files
wiki_files = list(DATA_DIR.glob("wiki_*.json"))
print(f"\nFound {len(wiki_files)} Wikipedia files to process")

input("\nPress ENTER to continue or CTRL+C to cancel...")

print("\n[SETUP] Starting initial load...\n")

try:
    load_json_files()
    print("\n" + "=" * 60)
    print("✓ Setup complete!")
    print("=" * 60)
    print("\nYour vector database is ready. Future startups will be instant.")
    print("You can now run: python3 walle_voice.py")
except Exception as e:
    print(f"\n[ERROR] Setup failed: {e}")
    sys.exit(1)
