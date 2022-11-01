import os
import json

with open("Realignment_MOS.csv", "r") as f:
    lines = list(f.readlines())

data = {}

for i, line in enumerate(lines):
    if i == 0:
        continue
    if line == ";;;;;;;\n":
        continue
    words = list(line.split(";"))
    case = words[1]
    mos = float(words[2].replace(",", "."))

    data[case] = mos

with open("parsed_mos.json", "w") as f:
    json.dump(data, f)