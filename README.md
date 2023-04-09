

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

the final_results has the personality types identified with the weightage . The results may look like the below  

  

    python example.py
     Type : ENTJ , Percentage : 16 , Characteristics : Strategic, logical, efficient, outgoing, ambitious, independent Effective organizers of people and long-range planners
     Type : ESTJ , Percentage : 11 , Characteristics : Efficient, outgoing, analytical, systematic, dependable, realistic Like to run the show and get things done in an orderly fashion
     Type : INFJ , Percentage : 16 , Characteristics : Idealistic, organized, insightful, dependable, compassionate, gentle Seek harmony and cooperation, enjoy intellectual stimulation
     Type : INTJ , Percentage : 33 , Characteristics : Innovative, independent, strategic, logical, reserved, insightful Driven by their own original ideas to achieve improvements
     Type : INTP , Percentage : 11 , Characteristics : Intellectual, logical, precise, reserved, flexible, imaginative Original thinkers who enjoy speculation and creative problem solving
