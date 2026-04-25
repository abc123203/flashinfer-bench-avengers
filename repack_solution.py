import json
import os

with open("solution.json", "r") as f:
    data = json.load(f)

for s in data["sources"]:
    if s["path"] != "binding.py":
        with open(os.path.join("solution/cuda", s["path"]), "r") as f:
            s["content"] = f.read()

with open("solution.json", "w") as f:
    json.dump(data, f, indent=2)
print("Repacked solution.json with latest code from solution/cuda!")
