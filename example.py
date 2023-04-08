from pypersonality.personality import PyPersonality
from functools import reduce


def read_file_as_list(filename):
    list = []
    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            if not line.isspace():
                list.append(line)
    return list


p = PyPersonality()
dictlist = [{}]

data = read_file_as_list("demo_text/modi.txt")
for line in data:
    dictlist.append(p.get_personality(line))

final_results = {}
for dictionary in dictlist:
    for key, value in dictionary.items():
        if key in final_results:
            final_results[key] = value + final_results[key]
        else:
            final_results[key] = value

total = sum(final_results.values())
print("Personlity Profile from text")
for key, value in final_results.items():
    if value != 0:
        properties = p.describe_type(key)
        percentage = (value / total) * 100
        print("Type : %s , Percentage : %d , Characteristics : %s" % (key, percentage, properties))
