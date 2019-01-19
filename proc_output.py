# coding: utf-8

"""This was used to process the output of the experiments."""

with open(file_path) as file:
    i = 0
    gold, norm, nonorm = {}, {}, {}
    for line in file:
        trim = line.find('>>>>>>') + 7
        name_end = trim + line[trim:].find(' ')
        vec_pos = trim + line[trim:].find('=') + 2
        name = line[trim:name_end]
        if i % 4 != 0:
            vals = eval(line[vec_pos:])
        if i % 4 == 1:
            gold[name] = vals
        if i % 4 == 2:
            norm[name] = vals
        if i % 4 == 3:
            nonorm[name] = vals
        i += 1
        
