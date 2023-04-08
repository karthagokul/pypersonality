
# pypersonality

Written in python , Module to identify the personality type from a text given using (MBTI) Myers-Briggs Personality Type Dataset.

![](https://github.com/karthagokul/pypersonality/blob/main/personality_types.jpg?raw=true)

### How to use it?
Install the module from pypi https://pypi.org/project/pypersonality/

    pip install pypersonality

import the module

    from pypersonality.personality import PyPersonality

Now you have a text file source, open and read the strings ,read_File_as_list is a function which splits the text files into strings.

    dictlist = [{}]
    data = read_file_as_list("demo_text/modi.txt")
    for line in data:
	    dictlist.append(p.get_personality(line))

Now the dictlist list has all the results, May be we need to combine all the dicontry entries

    final_results = {}
    for dictionary in dictlist:
    for key, value in dictionary.items():
    if key in final_results:
    final_results[key] = value + final_results[key]
    else:
    final_results[key] = value

the final_results has the personality types identified with the weightage