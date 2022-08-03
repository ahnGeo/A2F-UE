# @Geo Ahn
# Mapping (live link face app csv file) arkit blendshape to unreal metahuman animation curve
import pickle as pc

with open("./mod_blendshape_mapping.txt") as f :
    read_lines = f.readlines()

curves_name_list = set()

curves_value_name = []
idx = 0
for line in read_lines :
    line_split = line.split()
    if line_split[0][0] == '0' :
        curve_value = float(line_split[0])
        for i in range(1, len(line_split)) :
            value_name = (curve_value, line_split[i])
            curves_value_name[idx - 1].append(value_name)
    else :
            curves_value_name.append([])
            for i in range(1, len(line_split)) :
                if len(line_split[i]) <= 2 :
                    continue
                value_name = (1, line_split[i])
                curves_value_name[idx].append(value_name)
                curves_name_list.add(line_split[i])
            idx = idx + 1

with open("./curves_list_" + "audio_name" + "_mh_arkit.pkl","wb") as f:
    pc.dump(curves_value_name, f)

# curves_name_list = list(curves_name_list)
# print(curves_name_list)