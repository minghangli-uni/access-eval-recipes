"""
ESMF Profiling tool
The ESMF Performance tool is a Python-based tool designed to read and process 
performance profile data from ESMF profiling log files. It provides a 
structured way to extract hierachical timing and computational stats for 
various regions within ESMF runs, enabling detailed performance analysis.

 - esmfRunTimeCollection.py: handles the input ESMF profile files and constructs 
 the hierarchical data structure from the log entries.
 - esmfFileParser.py: handles the input ESMF profile files and constructs 
 the hierarchical data structure from the log entries.
 - esmfRegion.py: defines the ESMFRegion class, which represents individual
 regions of the ESMF performance data.

Latest version: xxx
Author: Minghang Li
Email: minghang.li1@anu.edu.au
License: Apache 2.0 License http://www.apache.org/licenses/LICENSE-2.0.txt
"""


# ===========================================================================
import os
import re
from esmfRegion import esmfRegion

def list_esmf_files(dir_path, esmf_summary, profile_prefix, summary_profile):
    """Lists ESMF files based on a prefix."""
    files = os.listdir(dir_path)
    if not esmf_summary:
        # ESMF_Profile.xxxx
        matching_files = [file for file in files if file.startswith(profile_prefix) and file != summary_profile]
        matching_files.sort(key=lambda x: int(x[len(profile_prefix):]))
    else:
        # ESMF_Profile.summary
        matching_files = [summary_profile]

    matching_files_path = [os.path.join(dir_path, matching_file) for matching_file in matching_files]
    return matching_files_path

def collect_total_runtime(ESMF_path, varnames=['[ESMF]'], profile_prefix='ESMF_Profile.', summary_profile='ESMF_Profile.summary', esmf_summary=True, index=2):
    runtime_tot = []
    for i in range(len(ESMF_path)):
        subfiles_path = list_esmf_files(ESMF_path[i], esmf_summary, profile_prefix, summary_profile)
        ESMF_region_all = []
        for subfile in subfiles_path:
            with open(subfile,'r') as file:
                eff_lines = []
                skip = False
                ESMF_region = build_ESMF_trees(file, skip, esmf_summary=esmf_summary)
                ESMF_region_all.append(ESMF_region)
    
        runtime = region_time_consumption(varnames, ESMF_region_all, index, esmf_summary)
        runtime_tot.append(runtime)
    return runtime_tot

def region_time_consumption(varnames, ESMF_region_all, index):
    """Calculates time consumption for specific regions."""
    runtime = {}
    for varname in varnames:
        runtime[varname] = [find_region_value(sub_ESMF_region, varname)[0][index] for sub_ESMF_region in ESMF_region_all]
    return runtime

def find_region_value(region, target_region, esmf_summary):
    """Recursively searches for a region value based on its hierarchical path."""
    target_parts = target_region.split('/')
    default_nans = (None,) * 6

    if not target_parts:
        return default_nans, False

    if region.name == target_parts[0]:
        if len(target_parts) == 1:
            if not esmf_summary:
                return (region.count, region.total, region.self_time, region.mean, region.min_time, region.max_time), True
            else:
                return (region.count, region.PETs, region.mean, region.min_time, region.max_time, region.min_PET, region.max_PET), True

        for child in region.children:
            result, found = find_region_value(child, '/'.join(target_parts[1:]))
            if found:
                return result, found

    for child in region.children:
        result, found = find_region_value(child, target_region)
        if found:
            return result, found

    return default_nans, False