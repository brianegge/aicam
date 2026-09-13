"""Per-class instance counts in the TEST split, by domain.

This is what decides whether the split can even be evaluated: if a class only
ever appears in one domain, the other specialist is never asked for it, and
the split costs nothing on this data -- but it also means that specialist
would fail completely the first time that class appears in the other domain.
"""
import json, os, sys
root = sys.argv[1]
stats = json.load(open(os.path.join(root, "colorstats.json")))
names = [l.strip("- \n") for l in open(os.path.join(root, "data.yaml")) if l.startswith("- ")]
tot = {}
for fname, hue, sat in stats["test"]:
    lbl = os.path.join(root, "test", "labels", fname.rsplit(".", 1)[0] + ".txt")
    if not os.path.exists(lbl):
        continue
    for line in open(lbl):
        if line.split():
            c = names[int(line.split()[0])]
            d = tot.setdefault(c, [0, 0])
            d[0 if hue else 1] += 1
print("%-9s %7s %7s %7s" % ("class", "color", "grey", "total"))
for c in names:
    col, g = tot.get(c, [0, 0])
    print("%-9s %7d %7d %7d" % (c, col, g, col + g))
