# Quick Data Folder

This folder is for adding custom JSON data to WALL-E's knowledge base.

## How to Use

1. **Add your JSON files here** - Put any `.json` file in this folder
2. **Run the loader** - Execute: `python load_quick_data.py`
3. **Done!** - WALL-E will now know the data

## Workflow

Whenever you add a new JSON file:
```bash
# Step 1: Add your file to quick_data/
# (copy or create your .json file here)

# Step 2: Load it into WALL-E
python load_quick_data.py

# Step 3: Talk to WALL-E
python walle_rag.py
```

## JSON Format

Any valid JSON structure works! Examples:

**Simple:**
```json
{
  "name": "Eliya",
  "age": 25,
  "city": "New York"
}
```

**Nested:**
```json
{
  "person": {
    "name": "Eliya",
    "favorites": {
      "food": "Chicken Wings",
      "game": "League of Legends"
    }
  }
}
```

**Lists:**
```json
{
  "hobbies": ["Gaming", "Coding", "Robotics"],
  "skills": ["Python", "JavaScript", "AI"]
}
```

All data will be flattened and made searchable for WALL-E!
