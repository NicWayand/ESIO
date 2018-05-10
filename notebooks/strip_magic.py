'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

import sys

# Need one input
if len(sys.argv) != 2:
    raise ValueError('Requires one arguments [notebook file]')

# Get name of configuration file/module
nbconvert_file = sys.argv[1]

# These functions were provided by https://github.com/rgerkin
# https://github.com/jupyter/nbconvert/issues/503
def strip_line_magic(line,magics_allowed):
        import re
        matches = re.findall("run_line_magic\(([^]]+)", line)
        if matches and matches[0]: # This line contains the pattern
            match = matches[0]
            if match[-1] == ')':
                match = match[:-1] # Just because the re way is hard
            magic_kind,stripped = eval(match)
            if magic_kind not in magics_allowed:
                stripped = "" # If the part after the magic won't work, just get rid of it
        else:
            #print("No line magic pattern match in '%s'" % line)
            stripped = line
        return stripped

def strip_magic(file_in):   
    magics_allowed = []
    # Read in
    with open(file_in) as f:
        code = f.read()
    code = code.split('\n')
    # Strip out magic
    n_code = []
    for cl in code:
        n_code.append(strip_line_magic(cl,magics_allowed))
    # Overwrite orig file
    with open(file_in,'w') as fo:
        for item in n_code:
            fo.write("%s\n" % item)

# Call Function
strip_magic(nbconvert_file)

