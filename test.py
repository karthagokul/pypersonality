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

new_dictionary = {}
for dictionary in dictlist:
    for key, value in dictionary.items():
        if key in new_dictionary:
            new_dictionary[key] = value + new_dictionary[key]
        else:
            new_dictionary[key] = value

total = sum(new_dictionary.values())
print("Personlity Profile from text")
for key, value in new_dictionary.items():
    if value != 0:
        properties = p.describe_type(key)
        percentage = (value / total) * 100
        print("Type : %s , Percentage : %d , Characteristics : %s" % (key, percentage, properties))
