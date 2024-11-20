import sys
import os
import re
from pprint import pprint

try:
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True
except ImportError:
    print("\nFatal error: modules not available.")
    print("On NCI, do the following and try again:")
    print("   module use /g/data/vk83/modules && module load payu/1.1.5\n")
    raise

def _read_ryaml(yaml_path):
    """ Read yaml file and preserve comments"""
    with open(yaml_path, "r") as f:
        return ryaml.load(f)

def _expt_fullname(nested_dict):
    num_combo = len(next(iter(nested_dict.values())))
    strings = []
    for i in range(num_combo):
        tmp = []
        for k,vs in nested_dict.items():
            v = vs[i]
            if isinstance(v,float) and v.is_integer():
                v = int(v)
            tmp.append(f"{k}_{v}")
        strings.append("_".join(tmp))
    return strings

def _extract_ntasks_values(expt_fullnames):
    results = []
    pattern = re.compile(r'(atm|cpl|ice|ocn|rof)_ntasks_(\d+)|(ocn)_rootpe_(\d+)')
    for expt in expt_fullnames:
        task_dict = {}
        matches = pattern.findall(expt)
        for match in matches:
            if match[0]:
                task_dict[f'{match[0]}_ntasks'] = int(match[1])
            if match[2]:
                task_dict[f'{match[2]}_rootpe'] = int(match[3])
        results.append(task_dict)
    return results

def _extract_restart_stop_values(expt_fullnames):
    results = []
    pattern = re.compile(r'restart_n_(\d+)_restart_option_(\w+)_stop_n_(\d+)_stop_option_(\w+)')
    for expt in expt_fullnames:
        match = pattern.search(expt)
        if match:
            result = {
                'restart_n': int(match.group(1)),
                'restart_option': match.group(2),
                'stop_n': int(match.group(3)),
                'stop_option': match.group(4)
            }
            results.append(result)
    
    return results

def list_esmf_files(dir_path, start_prefix, esmf_summary=False, summary_profile = 'ESMF_Profile.summary'):
    files = os.listdir(dir_path)
    if not esmf_summary:
        matching_files = [file for file in files if file.startswith(start_prefix) and file!=summary_profile]
        matching_files.sort(key=lambda x:int(x[len(start_prefix):]))
        matching_files_path = [os.path.join(dir_path, matching_file) for matching_file in matching_files]
    else:
        matching_file = summary_profile
        matching_files_path = [os.path.join(dir_path, matching_file)]
    return matching_files_path

def parse_line(line, esmf_summary=False):
    parts = re.split(r'\s{2,}',line.strip())
    if not esmf_summary:
        name = parts[0]
        count = int(parts[1])
        total = float(parts[2])
        self_time = float(parts[3])
        mean = float(parts[4])
        min_time = float(parts[5])
        max_time = float(parts[6])
        collect_data = (name, count, total, self_time, mean, min_time, max_time)
    else:
        name = parts[0]
        PETs = int(parts[1])
        PEs = int(parts[2])
        count = int(parts[3])
        mean = float(parts[4])
        min_time = float(parts[5])
        min_PET = int(parts[6])
        max_time = float(parts[7])
        max_PET = int(parts[8])
        collect_data = (name, count, PETs, PEs, mean, min_time, max_time, min_PET, max_PET)
    return collect_data

class ESMFRegion(object):
    def __init__(self, collect_data, esmf_summary=False):
        if not esmf_summary:
            self.name = collect_data[0]
            self.count = collect_data[1]
            self.total = collect_data[2]
            self.self_time = collect_data[3]
            self.mean = collect_data[4]
            self.min_time = collect_data[5]
            self.max_time = collect_data[6]
        else:
            self.name = collect_data[0]
            self.count = collect_data[1]
            self.PETs = collect_data[2]
            self.PEs = collect_data[3]
            self.mean = collect_data[4]
            self.min_time = collect_data[5]
            self.max_time = collect_data[6]
            self.min_PET = collect_data[7]
            self.max_PET = collect_data[8]

        self.children = []
        self.esmf_summary = esmf_summary

    def add_child(self,child):
        self.children.append(child)

    def to_dict(self):
        if not self.esmf_summary:
            profile_dict = {
                'name':self.name,
                'count': self.count,
                'total': self.total,
                'self_time': self.self_time,
                'mean': self.mean,
                'min_time': self.min_time,
                'max_time': self.max_time,
                'children': [child.to_dict() for child in self.children]
            }
        else:
            profile_dict = {
                'name':self.name,
                'count': self.count,
                'PETs': self.PETs,
                'PEs': self.PEs,
                'mean': self.mean,
                'min_time': self.min_time,
                'max_time': self.max_time,
                'min_PET': self.min_PET,
                'max_PET': self.max_PET,
                'children': [child.to_dict() for child in self.children]
            }
        return profile_dict

def build_ESMF_trees(lines, skip, esmf_summary=False):
    stack = []
    esmf_region = None
    for line in lines:
        if not line.strip():
            pass
        if line.startswith('  [ESMF]'):
            skip=True
        if skip:
            # determine hierachy
            indent_level = len(line) - len(line.lstrip())
            collect_data = parse_line(line, esmf_summary)
            region = ESMFRegion(collect_data, esmf_summary)
            if not stack:
                # the first esmf_region
                esmf_region = region
            else:
                while stack and stack[-1][1] >= indent_level:
                    stack.pop()
                if stack:
                    stack[-1][0].add_child(region)
            stack.append((region,indent_level))
    return esmf_region

def tree_to_dict(region):
    return region.to_dict()

def find_region_value(region, target_region, esmf_summary):
    """Recursively search for a region based on a full hierarchical path."""
    target_parts = target_region.split('/')
    default_nans = (None,)*6
    if not target_parts:
        return default_nans, False

    
    # Check if the current region matches the first part in the hierarchy
    if region.name == target_parts[0]:
        # If the full hierarchy is found, check if we need to go deeper
        if len(target_parts) == 1:
            # We found the region at the end of the path
            if not esmf_summary:
                # collect_data = (name, count, total, self_time, mean, min_time, max_time)
                return (region.count, region.total, region.self_time, region.mean, region.min_time, region.max_time), True
            else:
                # collect_data = (name, count, PETs, PEs, mean, min_time, max_time, min_PET, max_PET)
                return (region.count, region.PETs, region.mean, region.min_time, region.max_time, region.min_PET, region.max_PET), True
        
        for child in region.children:
            result, found = find_region_value(child, '/'.join(target_parts[1:]), esmf_summary)
            if found:
                return result, found

    # Continue searching through children if not found
    for child in region.children:
        result, found = find_region_value(child, target_region, esmf_summary)
        if found:
            return result, found

    return default_nans, False

def region_time_consumption(varnames, esmf_region_all, index, esmf_summary):
    runtime = {}
    for varname in varnames:
        runtime[varname] = [find_region_value(sub_esmf_region, varname, esmf_summary)[0][index] for sub_esmf_region in esmf_region_all]
    return runtime

def collect_runtime_tot(ESMF_path, varnames=['[ESMF]'], start_prefix='ESMF_Profile.', esmf_summary=True, index=2):
    runtime_tot = []
    for i in range(len(ESMF_path)):
        subfiles_path = list_esmf_files(ESMF_path[i], start_prefix, esmf_summary=esmf_summary, summary_profile = 'ESMF_Profile.summary')
        esmf_region_all = []
        for subfile in subfiles_path:
            with open(subfile,'r') as file:
                eff_lines = []
                skip = False
                esmf_region = build_ESMF_trees(file, skip, esmf_summary=esmf_summary)
                esmf_region_all.append(esmf_region)
    
        runtime = region_time_consumption(varnames, esmf_region_all, index, esmf_summary)
        runtime_tot.append(runtime)
    return runtime_tot






