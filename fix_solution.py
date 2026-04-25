import json
with open("solution.json", "r") as f:
    data = json.load(f)

keep = ["binding.py", "kernel.cu", "kernel.h", "main.cpp"]
data["sources"] = [s for s in data["sources"] if s["path"] in keep]

with open("solution.json", "w") as f:
    json.dump(data, f, indent=2)
print("solution.json updated!")
