#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add personal facts to WALL-E's knowledge base.
This makes WALL-E remember things about you!

Usage:
    python add_personal_data.py personal_facts.json
"""

import sys
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# CONFIG
BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "vector_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def add_personal_facts(json_path: Path):
    """Add personal facts from JSON to vector database."""
    try:
        print(f"\n[LOADING] {json_path.name}...")

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract facts from comprehensive profile
        facts = []

        # Personal info
        if "personal_info" in data:
            info = data["personal_info"]
            if "full_name" in info:
                facts.append(f"The user's full name is {info['full_name']}")
            if "nickname" in info:
                facts.append(f"The user's nickname is {info['nickname']}")
            if "birthday" in info:
                facts.append(f"The user's birthday is {info['birthday']}")
            if "age" in info:
                facts.append(f"The user is {info['age']} years old")
            if "current_residence" in info:
                facts.append(f"The user lives in {info['current_residence']}")
            if "birthplace" in info:
                facts.append(f"The user was born in {info['birthplace']}")

        # Appearance
        if "appearance" in data:
            for key, value in data["appearance"].items():
                facts.append(f"The user's {key.replace('_', ' ')} is {value}")

        # Preferences (handle lists and strings)
        if "preferences" in data:
            for key, value in data["preferences"].items():
                if isinstance(value, list):
                    value_str = ", ".join(str(v) for v in value)
                    facts.append(f"The user's favorite {key.replace('_', ' ')} are: {value_str}")
                else:
                    facts.append(f"The user's favorite {key.replace('_', ' ')} is {value}")

        # Personality
        if "personality" in data:
            for key, value in data["personality"].items():
                if isinstance(value, list):
                    facts.append(f"The user's {key.replace('_', ' ')}: {', '.join(value)}")
                else:
                    facts.append(f"The user's {key.replace('_', ' ')}: {value}")

        # Health & Fitness
        if "health_fitness" in data:
            for key, value in data["health_fitness"].items():
                if isinstance(value, list):
                    facts.append(f"The user's {key.replace('_', ' ')}: {', '.join(value)}")
                else:
                    facts.append(f"The user's {key.replace('_', ' ')}: {value}")

        # Professional life
        if "professional_life" in data:
            for key, value in data["professional_life"].items():
                if isinstance(value, list):
                    facts.append(f"The user's {key.replace('_', ' ')}: {', '.join(value)}")
                else:
                    facts.append(f"The user's {key.replace('_', ' ')}: {value}")

        # Social relationships
        if "social_relationships" in data:
            social = data["social_relationships"]
            if "partner" in social:
                facts.append(f"The user's partner is {social['partner']}")
            if "friends" in social:
                facts.append(f"The user's friends include: {', '.join(social['friends'])}")
            if "family" in social:
                for relation, name in social["family"].items():
                    facts.append(f"The user's {relation} is {name}")
            if "pets" in social:
                for pet in social["pets"]:
                    facts.append(f"The user had a pet named {pet.get('name', 'Unknown')}, a {pet.get('type', 'pet')}")

        # Goals
        if "goals_aspirations" in data:
            for key, value in data["goals_aspirations"].items():
                if isinstance(value, list):
                    facts.append(f"The user's {key.replace('_', ' ')}: {', '.join(value)}")
                else:
                    facts.append(f"The user's {key.replace('_', ' ')}: {value}")

        # Hobbies (in miscellaneous)
        if "miscellaneous" in data and "hobbies" in data["miscellaneous"]:
            hobbies = data["miscellaneous"]["hobbies"]
            facts.append(f"The user's hobbies are: {', '.join(hobbies)}")

        if not facts:
            print(f"  [ERROR] No facts found in {json_path.name}")
            return 0

        # Load vector DB
        print(f"  [CONNECTING] Loading vector database...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectordb = Chroma(
            persist_directory=str(VECTOR_DB_DIR),
            embedding_function=embeddings
        )

        # Add to vector DB
        print(f"  [ADDING] {len(facts)} personal facts to WALL-E's memory...")
        vectordb.add_texts(
            facts,
            metadatas=[{"source": "personal_data", "type": "user_info"}] * len(facts)
        )

        print(f"  [OK] Successfully taught WALL-E {len(facts)} things about you!")
        print("\nFacts added:")
        for fact in facts:
            print(f"  - {fact}")

        return len(facts)

    except Exception as e:
        print(f"  [ERROR] Failed to process {json_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    print("="*60)
    print("WALL-E Personal Data Trainer")
    print("="*60)
    print("\nThis teaches WALL-E facts about you!")
    print("Edit personal_facts.json with your information.\n")

    # Get JSON file
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
        # Make it absolute if it's relative
        if not json_path.is_absolute():
            json_path = BASE_DIR / json_path
    else:
        json_path = BASE_DIR / "personal_facts.json"

    if not json_path.exists():
        print(f"[ERROR] File not found: {json_path}")
        print(f"\nCreate a file called '{json_path.name}' with your personal info.")
        print("See personal_facts.example.json for the format.")
        return

    # Process file
    facts_added = add_personal_facts(json_path)

    if facts_added > 0:
        print("\n" + "="*60)
        print("[SUCCESS] WALL-E now knows more about you!")
        print("Try asking: 'When is my birthday?' or 'What are my hobbies?'")
        print("="*60)

if __name__ == "__main__":
    main()
