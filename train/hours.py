"""When does each class actually appear? Uses the HHMMSS filename prefix and
the real label file, across every split -- the whole labelled history, not one
split.
"""
import glob, json, os, sys, collections
root = sys.argv[1]
names = [l.strip("- \n") for l in open(os.path.join(root, "data.yaml")) if l.startswith("- ")]
stats = {}
for s in ("train", "valid", "test"):
    for fn, hue, sat in json.load(open(os.path.join(root, "colorstats.json")))[s]:
        stats[fn] = (s, hue)
byhour = collections.defaultdict(lambda: collections.defaultdict(int))
domain = collections.defaultdict(lambda: [0, 0])
for fn, (split, hue) in stats.items():
    if not fn[:6].isdigit():
        continue
    hh = int(fn[:2])
    lbl = os.path.join(root, split, "labels", fn.rsplit(".", 1)[0] + ".txt")
    if not os.path.exists(lbl):
        continue
    for line in open(lbl):
        f = line.split()
        if len(f) >= 5:
            c = names[int(f[0])]
            byhour[c][hh] += 1
            domain[c][0 if hue else 1] += 1
print("%-9s %6s %6s   %s" % ("class", "colour", "grey", "instances by hour (00..23)"))
for c in names:
    col, g = domain[c]
    bars = " ".join("%3d" % byhour[c].get(h, 0) for h in range(24))
    print("%-9s %6d %6d   %s" % (c, col, g, bars))
print()
print("colour-domain instances outside 20:00-06:00 (i.e. genuinely daytime):")
for c in names:
    day = sum(v for h, v in byhour[c].items() if 6 <= h < 20)
    print("  %-9s daytime-hour instances: %d" % (c, day))
