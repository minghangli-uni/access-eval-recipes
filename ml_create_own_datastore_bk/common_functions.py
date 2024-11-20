import os
import sys
import re
import copy
import subprocess
import shutil
import glob
import argparse
import warnings
from datetime import datetime
try:
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import git
    import f90nml
    import cftime
    import xarray as xr
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True

    import pandas as pd
except ImportError:
    print("\nFatal error: modules not available.")
    print("On NCI, do the following and try again:")
    print("   module use /g/data/vk83/modules && module load payu/1.1.5\n")
    raise

def _set_default_plt_params():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (15, 10)
    })

def _read_ryaml(yaml_path):
    """ Read yaml file and preserve comments"""
    with open(yaml_path, "r") as f:
        return ryaml.load(f)

def _expt_name_combo(nested_dict):
    
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


def plot_time_series_combo(datastore_tot,MOM_names,var,indices=None,
                          line_style=None, line_color=None, line_marker=None,
                          figsize=(15, 10), tick_fontsize=12, label_fontsize=14,
                          subplots=None, difference=None,time_range=None,ylims=None,legend_loc='best',
                          file_save=None):
    
    # legend: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    def set_time_range(dataset, var, time_range):
        if time_range:
            start,end=time_range
            return dataset[var].sel(time=slice(start, end))
        return dataset[var]
        
    if indices is None:
        indices = range(len(datastore_tot))
    if line_style is None:
        line_style = ['-']*len(indices)
    if line_color is None:
        line_color = ['k']*len(indices)
    if line_marker is None:
        line_marker = ['']*len(indices)
        
    plt.figure(figsize=figsize)

    if subplots:
        for _subplot in subplots:
            nrows,ncols,index = _subplot['subplot']
            indices = _subplot['indices']
            title = _subplot['title']
            difference = _subplot['difference']
            ylim = _subplot.get('ylim',None)
            legend_loc = _subplot.get('legend_loc','best')

            plt.subplot(nrows,ncols,index)

            if difference is not None:
                base_index = difference
                comp_indice = [j for j in indices if j!=base_index]
                base_tmp = datastore_tot[base_index].search(variable=var)
                base_dict = base_tmp.to_dataset_dict(progressbar=False)
                base_dataset = base_dict[next(iter(base_tmp))]
                base_data = set_time_range(base_dataset,var,time_range)
                for j in comp_indice:
                    comp_tmp = datastore_tot[j].search(variable=var)
                    comp_dict = comp_tmp.to_dataset_dict(progressbar=False)
                    comp_dataset = comp_dict[next(iter(comp_tmp))]
                    comp_data = set_time_range(comp_dataset,var,time_range)

                    # difference
                    diff = abs(comp_dataset[var] - base_dataset[var])
                    plt.plot(diff.time, diff, 
                             label=f"Difference (base: {MOM_names[base_index]}\n comp: {MOM_names[j]})",
                                 linestyle = line_style[j%len(line_style)],
                                 color = line_color[j%len(line_color)],
                                 marker = line_marker[j%len(line_marker)])
                if ylim is not None:
                    plt.ylim(ylim)
                plt.title(title)
                plt.legend(loc=legend_loc)
                plt.grid()
                
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
                plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)
            
            else:
                for i in indices:
                    datastore_tmp = datastore_tot[i].search(variable=var)
                    dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
                    dataset = dataset_dict[next(iter(dataset_dict))]
                    data = set_time_range(dataset, var, time_range)
                    data.plot(label = MOM_names[i],
                                     linestyle = line_style[i%len(line_style)],
                                     color = line_color[i%len(line_color)],
                                     marker = line_marker[i%len(line_marker)])
                if ylims is not None:
                    plt.ylim(ylims)
                plt.title(f"Comparison of {var}")
                plt.legend(loc=legend_loc)
                plt.grid()
            
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
                plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)
    
    else:
        for i in indices:
            datastore_tmp = datastore_tot[i].search(variable=var)
            dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
            dataset = dataset_dict[next(iter(dataset_dict))]
            data = set_time_range(dataset, var, time_range)
            data.plot(label = MOM_names[i],
                             linestyle = line_style[i%len(line_style)],
                             color = line_color[i%len(line_color)],
                             marker = line_marker[i%len(line_marker)])
        if ylims is not None:
            plt.ylim(ylims)
        plt.title(f"Comparison of {var}")
        plt.legend(loc=legend_loc)
        plt.grid()
    
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
        plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)

    plt.tight_layout()
    plt.show()

def plot_time_series_combo3(datastore_tot, MOM_names,var, indices=None, base_index=None,
                          line_style=None, line_color=None, line_marker=None, markersize=None, line_width=None,
                          figsize=(15,10), tick_fontsize=20, label_fontsize=20, title_fontsize=20,
                          legend_loc='best',legend_fontsize=20,
                          plots_config=None, use_subplots=False, nrows=None, ncols=2,
                          time_range=None,
                          xlims=None, num_intervals=None, ylims=None,
                          file_path = None):

    def dict_data_and_sorted_keys(data_store, var):
        datastore_tmp = data_store.search(variable=var)
        dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
        sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
        return dataset_dict, sorted_keys

    def plot_ticks_labels():
        if ylims is not None:
            ax.set_ylim(ylims)
        if xlims is not None:
            xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
            ax.set_xlim(xlims_pd)
            if num_intervals is not None:
                tick_positions = np.linspace(start=xlims_pd[0].value, stop=xlims_pd[1].value, num=num_intervals + 2)
                tick_positions = pd.to_datetime(tick_positions)
                ax.set_xticks(tick_positions)
                custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
                ax.set_xticklabels(custom_labels)
        ax.set_title(title, fontsize=title_fontsize)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
# legend: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

    def set_time_range(dataset, var, time_range):
        if time_range is not None:
            start,end = time_range
            return dataset[var].sel(time=slice(start, end))
        return dataset[var]

    def convert_cftime_to_datetime(cftime_array):
        datetime_list = []
        for dt in cftime_array:
            dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                       hour=dt.hour, minute=dt.minute, second=dt.second)
            datetime_list.append(dt_datetime)
        return np.array(datetime_list)

    if plots_config:
        indices = plots_config[0].get('indices', list(range(len(datastore_tot))))
    else:
        indices = list(range(len(datastore_tot)))

    if use_subplots:
        n_data = len(indices)
        if nrows is None:
            nrows = (n_data + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if (nrows * ncols > 1) else [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    line_style = ['-'] * len(indices) if line_style is None else line_style
    line_color = ['k'] * len(indices) if line_color is None else line_color
    line_marker = [''] * len(indices) if line_marker is None else line_marker
    line_width = [5] * len(indices) if line_width is None else line_width
    markersize = [0] * len(indices) if markersize is None else markersize

    for plot_indx, _subplot in enumerate(plots_config):
        title = _subplot.get('title', '')
        base_index = _subplot.get('base_index', None)
        if base_index is not None:
            base_dict, sorted_base_keys = dict_data_and_sorted_keys(datastore_tot[base_index], var)

            for tmp_indx, i in enumerate(indices):
                if use_subplots:
                    ax = axes[tmp_indx]
                dataset_dict, sorted_keys = dict_data_and_sorted_keys(datastore_tot[i], var)

                for j, key in enumerate(sorted_keys):
                    dataset = dataset_dict[key]
                    data = set_time_range(dataset, var, time_range)
                    data['time'] = convert_cftime_to_datetime(data['time'].values)
                    base_data = base_dict.get(sorted_base_keys[j], None)
                    if base_data is not None:
                        base_data_aligned = set_time_range(base_data, var, time_range)
                        base_data_aligned['time'] = convert_cftime_to_datetime(base_data_aligned['time'].values)

                        data_diff = data - base_data_aligned
                        if not data_diff.isnull().all():
                            data_diff.plot(ax=ax,
                                           linestyle=line_style[i % len(line_style)],
                                           color=line_color[i % len(line_color)],
                                           marker=line_marker[i % len(line_marker)],
                                           markersize=markersize[j % len(markersize)],
                                           linewidth=line_width[j % len(line_width)],
                                           label=f"{MOM_names[i]} - Base" if j == 0 else "")
                            plot_ticks_labels()
                    else:
                        print(f"Base dataset for {key} not found.")
        else:
            for tmp_indx, i in enumerate(indices):
                if use_subplots:
                    ax = axes[tmp_indx]
                dataset_dict, sorted_keys = dict_data_and_sorted_keys(datastore_tot[i], var)

                for j, key in enumerate(sorted_keys):
                    dataset = dataset_dict[key]
                    data = set_time_range(dataset, var, time_range)
                    data['time'] = convert_cftime_to_datetime(data['time'].values)

                    if not data.isnull().all():
                        data.plot(ax=ax,
                                  linestyle=line_style[i % len(line_style)],
                                  color=line_color[i % len(line_color)],
                                  marker=line_marker[i % len(line_marker)],
                                  markersize=markersize[j % len(markersize)],
                                  linewidth=line_width[j % len(line_width)],
                                  label=f"{MOM_names[i]}" if j == 0 else "")
                        plot_ticks_labels()
                    else:
                        pass

    if use_subplots:
        for j in range(n_data, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    #     for plot_indx, _subplot in enumerate(subplots):
    #         print(plot_indx)
    #         title = _subplot['title']
    #         base_index = _subplot.get('base_index', None)
    #         ax = axes[plot_indx]  # Select the correct subplot axis
    #         if base_index is not None:
    #             comp_indices = indices

    #             base_tmp = datastore_tot[base_index].search(variable=var)
    #             base_dict = base_tmp.to_dataset_dict(progressbar=False)
    #             sorted_base_keys = sorted(base_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
    
    #             for i in comp_indices:
    #                 datastore_tmp = datastore_tot[i].search(variable=var)
    #                 dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
    #                 sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))

    #                 for j, key in enumerate(sorted_keys):
    #                     dataset = dataset_dict[key]
    #                     data = set_time_range(dataset, var, time_range)
    #                     data['time'] = convert_cftime_to_datetime(data['time'].values)

    #                     base_data = base_dict.get(sorted_base_keys[j], None)
    
    #                     if base_data is not None:
    #                         base_data_aligned = set_time_range(base_data, var, time_range)
    #                         base_data_aligned['time'] = convert_cftime_to_datetime(base_data_aligned['time'].values)
    
    #                         data_diff = data - base_data_aligned
    
    #                         if not data_diff.isnull().all():
    #                             data_diff.plot(ax=ax,
    #                                 linestyle=line_style[i % len(line_style)],
    #                                 color=line_color[i % len(line_color)],
    #                                 marker=line_marker[i % len(line_marker)],
    #                                 markersize=markersize[j % len(markersize)],
    #                                 label=f"{MOM_names[i]} - Base" if j == 0 else "",  # Only label the first plot in the loop
    #                             )
    #                     else:
    #                         print(f"Base dataset for {key} not found.")
    
    #         else:
    #             for i in indices:
    #                 datastore_tmp = datastore_tot[i].search(variable=var)
    #                 dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
    #                 sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
    
    #                 for j, key in enumerate(sorted_keys):
    #                     dataset = dataset_dict[key]
    #                     data = set_time_range(dataset, var, time_range)
    #                     data['time'] = convert_cftime_to_datetime(data['time'].values)
    
    #                     if not data.isnull().all():
    #                         data.plot(ax=ax,
    #                             linestyle=line_style[i % len(line_style)],
    #                             color=line_color[i % len(line_color)],
    #                             marker=line_marker[i % len(line_marker)],
    #                             markersize=markersize[j % len(markersize)],
    #                             label=f"{MOM_names[i]}" if j == 0 else "",  # Only label the first plot in the loop
    #                         )
    #                     else:
    #                         pass

    #         if ylims is not None:
    #             ax.set_ylim(ylims)
    #         if xlims is not None:
    #             xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
    #             ax.set_xlim(xlims_pd)
    #             if num_intervals is not None:
    #                 tick_positions = np.linspace(start=xlims_pd[0].value,
    #                                              stop=xlims_pd[1].value,
    #                                              num=num_intervals + 2)
    #                 tick_positions = pd.to_datetime(tick_positions)
    #                 ax.set_xticks(tick_positions)
    #                 custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
    #                 ax.set_xticklabels(custom_labels)

    #         ax.set_title(title)
    #         ax.legend(loc=legend_loc)
    #         ax.grid()
    #         # ax.set_xticks(fontsize=tick_fontsize)
    #         # ax.set_yticks(fontsize=tick_fontsize)
    #         ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)  # Set tick font size for both axes
    #         ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
    #         ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
    
    #     for j in range(len(subplots), len(axes)):
    #         print(j)
    #         axes[j].axis('off')

    # plt.tight_layout()
    # plt.show()



def plot_time_series_combo2(datastore_tot,MOM_names,var,indices=None,
                          line_style=None, line_color=None, line_marker=None, markersize=None,
                          figsize=(15,10), tick_fontsize=12, label_fontsize=12,
                          subplots=None, difference=None, time_range=None,
                          xlims=None, num_intervals=None, ylims=None, legend_loc='best',
                          file_path = None):
    # legend: supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    def set_time_range(dataset, var, time_range):
        if time_range is not None:
            start,end = time_range
            return dataset[var].sel(time=slice(start, end))
        return dataset[var]

    def cftime_to_date(cftime_dates):
        return [datetime(date.year, date.month, date.day) for date in cftime_dates]

    def convert_cftime_to_datetime(cftime_array):
        return np.array([dt.to_datetime() for dt in cftime_array])

    def convert_cftime_to_datetime(cftime_array):
        datetime_list = []
        for dt in cftime_array:
            dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                        hour=dt.hour, minute=dt.minute, second=dt.second)
            datetime_list.append(dt_datetime)
        return np.array(datetime_list)

    if indices is None:
        indices = list(range(len(datastore_tot)))

    if line_style is None:
        line_style = ['-']*len(indices)
    if line_color is None:
        line_color = ['k']*len(indices)
    if line_marker is None:
        line_marker = ['']*len(indices)
    if markersize is None:
        markersize = [0]*len(indices)
    plt.figure(figsize=figsize)

    if subplots:
        for _subplot in subplots:
            nrows,ncols,index = _subplot['subplot']
            if _subplot['indices'] is None:
                indices = list(range(len(datastore_tot)))
            else:
                indices = _subplot['indices']
            title = _subplot['title']
            difference = _subplot['difference']
            ylim = _subplot.get('ylim',None)
            legend_loc = _subplot.get('legend_loc','best')

            plt.subplot(nrows,ncols,index)

            if difference is not None:
                base_index = difference
                #comp_indices = [j for j in indices if j != base_index]
                comp_indices = indices

                # Load and sort base dataset
                base_tmp = datastore_tot[base_index].search(variable=var)
                base_dict = base_tmp.to_dataset_dict(progressbar=False)
                sorted_base_keys = sorted(base_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
                
                for i in comp_indices:  # Compare datasets in comp_indices against base
                    datastore_tmp = datastore_tot[i].search(variable=var)
                    dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
                    sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
                    
                    for j, key in enumerate(sorted_keys):
                        dataset = dataset_dict[key]
                        data = set_time_range(dataset, var, time_range)
                        data['time'] = convert_cftime_to_datetime(data['time'].values)
                        
                        # Get the corresponding base dataset for the same key
                        base_data = base_dict.get(sorted_base_keys[j], None)
                        
                        if base_data is not None:
                            base_data_aligned = set_time_range(base_data, var, time_range)
                            base_data_aligned['time'] = convert_cftime_to_datetime(base_data_aligned['time'].values)
                            
                            # Calculate the difference
                            data_diff = data - base_data_aligned
                            
                            if not data_diff.isnull().all():
                                data_diff.plot(
                                    linestyle=line_style[i % len(line_style)],  # Style based on the index i
                                    color=line_color[i % len(line_color)],      # Color based on the index i
                                    marker=line_marker[i % len(line_marker)],   # Marker based on the index i
                                    markersize=markersize[j % len(markersize)],
                                    label=f"{MOM_names[i]} - Base" if j == 0 else ""  # Only label the first plot in the loop
                                )
                            else:
                                pass
                        else:
                            print(f"Base dataset for {key} not found.")
                ax.set_title(title)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True)
                if ylims is not None:
                    ax.setylim(ylims)
                if xlims is not None:
                    xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
                    plt.xlim(xlims_pd)
                    if num_intervals is not None:
                        tick_positions = np.linspace(start = xlims_pd[0].value,
                                                     stop = xlims_pd[1].value,
                                                     num = num_intervals + 2)
                        tick_positions = pd.to_datetime(tick_positions)
                        plt.gca().set_xticks(tick_positions)
                        custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
                        plt.gca().set_xticklabels(custom_labels)
                plt.title(title)
                plt.legend(loc=legend_loc)
                plt.grid()

                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)
                plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
                plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)
            
            else:
                print(indices)
                for i in indices:
                    datastore_tmp = datastore_tot[i].search(variable=var)
                    dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
                    sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
                    
                    for j, key in enumerate(sorted_keys):
                        dataset = dataset_dict[key]
                        data = set_time_range(dataset, var, time_range)
                        data['time'] = convert_cftime_to_datetime(data['time'].values)
                        # print(data)
                        if not data.isnull().all():
                            data.plot(
                                linestyle=line_style[i % len(line_style)],  # Style based on the index i
                                color=line_color[i % len(line_color)],      # Color based on the index i
                                marker=line_marker[i % len(line_marker)],   # Marker based on the index i
                                markersize=markersize[j % len(markersize)],
                                label=MOM_names[i] if j == 0 else ""        # Only label the first plot in the loop
                            )
                        else:
                            pass

                if ylims is not None:
                    plt.ylim(ylims)
                if xlims is not None:
                    xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
                    plt.xlim(xlims_pd)
                    if num_intervals is not None:
                        tick_positions = np.linspace(start = xlims_pd[0].value,
                                                     stop = xlims_pd[1].value,
                                                     num = num_intervals + 2)
                        tick_positions = pd.to_datetime(tick_positions)
                        plt.gca().set_xticks(tick_positions)
                        custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
                        plt.gca().set_xticklabels(custom_labels)
                    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
                    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
                    # tick_positions = pd.date_range(start=xlims[0], end=xlims[1], freq='YS')
# pd.date_range():
# •	D: Day (calendar day)
# •	B: Business day
# •	M: Month end
# •	MS: Month start
# •	H: Hour
# •	T or min: Minute
# •	S: Second
# •	Y: Year end
# •	YS: Year start
                plt.title(f"Comparison of {var}")
                plt.legend(loc=legend_loc)
                plt.grid()
                plt.xticks(fontsize=tick_fontsize, rotation=45)
                plt.yticks(fontsize=tick_fontsize)
                plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
                plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)

    else:
        if difference is not None:
            base_index = difference
            comp_indice = [j for j in indices if j!=base_index]
            base_tmp = datastore_tot[base_index].search(variable=var)
            base_dict = base_tmp.to_dataset_dict(progressbar=False)
            sorted_base_keys = sorted(base_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
            base_data = {key: base_dict[key][var] for key in sorted_base_keys}
            for j in comp_indice:
                comp_tmp = datastore_tot[j].search(variable=var)
                comp_dict = comp_tmp.to_dataset_dict(progressbar=False)
                sorted_comp_keys = sorted(comp_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
                comp_data = {key: comp_dict[key][var] for key in sorted_comp_keys}

                for k, (base_key, comp_key) in enumerate(zip(sorted_base_keys, sorted_comp_keys)):
                    diff = abs(comp_data[comp_key][var] - base_data[base_key][var])
                    plt.plot(
                        diff.time, diff,
                        label=f"Difference (base: {MOM_names[base_index]}, comp: {MOM_names[j]}, {base_key} vs {comp_key})",
                        linestyle=line_style[j % len(line_style)],  # Use j here to keep consistent style for the index
                        color=line_color[j % len(line_color)],
                        marker=line_marker[j % len(line_marker)],
                        markersize=markersize[j % len(markersize)],
                    )
            if ylim is not None:
                plt.ylim(ylims)
            plt.title(title)
            plt.legend(loc=legend_loc)
            plt.grid()

            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
            plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)
        else:
            for i in indices:
                datastore_tmp = datastore_tot[i].search(variable=var)
                dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
                #dataset = dataset_dict[next(iter(dataset_dict))]
                #data = set_time_range(dataset, var, time_range)
                sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
                
                for j, key in enumerate(sorted_keys):
                    dataset = dataset_dict[key]
                    dataset[var].plot(
                        linestyle=line_style[i % len(line_style)],  # Style based on the index i
                        color=line_color[i % len(line_color)],      # Color based on the index i
                        marker=line_marker[i % len(line_marker)],   # Marker based on the index i
                        markersize=markersize[i % len(markersize)],
                        label=MOM_names[i] if j == 0 else ""        # Only label the first plot in the loop
                    )
            if ylims is not None:
                plt.ylim(ylims)
            plt.title(f"Comparison of {var}")
            plt.legend(loc=legend_loc)
            plt.grid()

            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.xlabel(plt.gca().get_xlabel(), fontsize=label_fontsize)
            plt.ylabel(plt.gca().get_ylabel(), fontsize=label_fontsize)

    plt.tight_layout()

    if file_path:
        base_path = os.path.dirname(file_path)
        os.makedirs(base_path, exist_ok=True)
        plt.savefig(file_path,dpi=300)

    plt.show()

def combine_datasets(dset):

    def get_time(dataset):
        if 'time' in dataset.coords:
            return dataset.time.values[0]
        else:
            raise ValueError(f"Dataset does not have a time coord: {dataset}")

    # create a list of tuples (time, dataset) for sorting
    dataset_w_time = [(get_time(dataset), dataset) for key, dataset in dset.items()]

    # sort datasets by time coords
    datasets_sorted = sorted(dataset_w_time, key=lambda x:x[0])

    # Extract only the sorted darrays
    darrays_sorted = [dataset for _, dataset in datasets_sorted]

    # Concatenate the sorted darrays along the time dim
    combined_data = xr.concat(darrays_sorted, dim='time')

    # Ensure time coords are ordered correctly
    combined_data = combined_data.sortby('time')

    return combined_data
def plot3d3(datastore_tot, MOM_names_tot: list, var: str, datastore_ctrl = None, depth_level = None, common_time = None,
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10)):

    if not isinstance(datastore_tot,list):
        datastore_tot = [datastore_tot]

    n_data = len(datastore_tot)

    if nrows is None:
        nrows = (n_data+ncols-1)//ncols

    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if datastore_ctrl is not None:
        cat_subset_contrl = datastore_ctrl.search(variable=[var])
        dset_contrl = cat_subset_contrl.to_dataset_dict(progressbar=False)
        combined_data_contrl = combine_datasets(dset_contrl)

        if common_time is not None:
            tmp_contrl = combined_data_contrl.sel(time=common_time,method='nearest')
        else:
            if time_selection == 'index':
                tmp_contrl = combined_data_contrl.isel(time=time_index)
            elif time_selection == 'date':
                tmp_contrl = combined_data_contrl.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp_contrl = combined_data_contrl.sel(time=slice(start_time,end_time))

    for _subplot, datastore in enumerate(datastore_tot):
        print(_subplot)
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(dset)
        #print(combined_data.time.values)
        #print(combined_data)
        if common_time is not None:
            tmp = combined_data.sel(time=common_time, method='nearest')
        else:
            if time_selection == 'index':
                tmp = combined_data.isel(time=time_index)
            elif time_selection == 'date':
                tmp = combined_data.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp = combined_data.sel(time=slice(start_time,end_time))

        if datastore_ctrl:
            data_plot = tmp[var] - tmp_contrl[var]
            title = f"diff - {MOM_names_tot[_subplot]}"
        else:
            data_plot = tmp[var]
            title = f"{MOM_names_tot[_subplot]}"
        date_info = data_plot.time.values
        title += f"\n{date_info}"

        ax = axes[_subplot]

        if len(data_plot.dims) == 3:
            data_plot = data_plot.isel(zl=depth_level)

        cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                             vmin = cbar_range[0] if cbar_range else None,
                             vmax = cbar_range[1] if cbar_range else None,
                             cmap='RdBu')
        if cbar.colorbar:
            cbar = cbar.colorbar
            cbar.ax.tick_params(labelsize=16)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(16)
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

    for j in range(n_data, len(axes)):
        axes[j].axis('off')     
    plt.tight_layout()
    plt.show()
    
def plot3d2(datastore_tot, MOM_names_tot: list, var: str, datastore_ctrl = None, depth_level = None, common_time = None,
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10)):
    
    if not isinstance(datastore_tot,list):
        datastore_tot = [datastore_tot]
        
    n_data = len(datastore_tot)

    if nrows is None:
        nrows = (n_data+ncols-1)//ncols

    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if datastore_ctrl is not None:
        cat_subset_contrl = datastore_ctrl.search(variable=[var])
        dset_contrl = cat_subset_contrl.to_dataset_dict(progressbar=False)
        key_contrl = list(dset_contrl.keys())
#        print(key_contrl)
        dset_sub_contrl = dset_contrl[key_contrl][var]

        if common_time is not None:
            tmp_contrl = dset_sub_contrl.sel(time=common_time,method='nearest')
        else:
            if time_selection == 'index':
                tmp_contrl = dset_sub_contrl.isel(time=time_index)
            elif time_selection == 'date':
                tmp_contrl = dset_sub_contrl.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp_contrl = dset_sub_contrl.sel(time=slice(start_time,end_time))
                tmp_contrl = tmp_contrl

    for _subplot, datastore in enumerate(datastore_tot):
        print(_subplot)
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(dset)
        print(combined_data.time.values)
        sorted_keys = sorted(dset.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
        dset_sub = {}
        for key in sorted_keys:
            dset_sub[key] = dset[key][var]

        print(dset_sub)
        if common_time is not None:
#            tmp = dset_sub.sel(time=common_time,method='nearest')
            # Create a new dictionary to store the results of the selection
            tmp = {}
            
            # Iterate over each key-value pair in dset_sub
            for key, dataset in dset_sub.items():
                print(key)
                print(dataset)
                try:
                    # Perform the .sel() operation on each Dataset or DataArray
                    tmp[key] = dataset.sel(time=common_time, method='nearest')
                    if datastore_ctrl:
                        data_plot = tmp - tmp_contrl
                        title = f"diff - {MOM_names_tot[_subplot]}"
                    else:
                        data_plot = tmp[key]
                        title = f"{MOM_names_tot[_subplot]}"
                except AttributeError:
                    print(f"Object under key {key} does not support .sel()")
        date_info = data_plot.time.values
        title += f"\n{date_info}"

        ax = axes[_subplot]

        if len(data_plot.dims) == 3:
            data_plot = data_plot.isel(zl=depth_level)
        
        cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                             vmin = cbar_range[0] if cbar_range else None,
                             vmax = cbar_range[1] if cbar_range else None,
                             cmap='RdBu')
        if cbar.colorbar:
            cbar = cbar.colorbar
            cbar.ax.tick_params(labelsize=16)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(16)

        

        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

    for j in range(n_data, len(axes)):
        axes[j].axis('off')     
    plt.tight_layout()
    plt.show()

def plot2d(datastore_tot, MOM_names_tot: list, var: str, datastore_ctrl = None, 
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10)):
    if not isinstance(datastore_tot,list):
        datastore_tot = [datastore_tot]
        
    n_data = len(datastore_tot)

    if nrows is None:
        nrows = (n_data+ncols-1)//ncols

    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if datastore_ctrl:
        cat_subset_contrl = datastore_ctrl.search(variable=[var])
        dset_contrl = cat_subset_contrl.to_dataset_dict(progressbar=False)
        key_contrl = list(dset_contrl.keys())[0]
        dset_sub_contrl = dset_contrl[key_contrl][var]

        if time_selection == 'index':
            tmp_contrl = dset_sub_contrl.isel(time=time_index)
        elif time_selection == 'date':
            tmp_contrl = dset_sub_contrl.sel(time=start_time)
        elif time_selection == 'date_slice':
            tmp_contrl = dset_sub_contrl.sel(time=slice(start_time,end_time))
            tmp_contrl = tmp_contrl
            
    for _subplot, datastore in enumerate(datastore_tot):
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)

        key = list(dset.keys())[0]
        dset_sub = dset[key][var]
        
        if time_selection == 'index':
            tmp = dset_sub.isel(time=time_index)
        elif time_selection == 'date':
            tmp = dset_sub.sel(time=start_time)
        elif time_selection == 'date_slice':
            tmp = dset_sub.sel(time=slice(start_time,end_time))

        if datastore_ctrl:
            data_plot = tmp - tmp_contrl
            title = f"diff - {MOM_names_tot[_subplot]}"
        else:
            data_plot = tmp
            title = f"{MOM_names_tot[_subplot]}"

        date_info = data_plot.time.values
        # print(type(date_info))
        # date_info = np.datetime64(date_info)
        # date_str = np.datetime_as_string(date_info, unit='hour')
        title += f"\n{date_info}"

        ax = axes[_subplot]
        cbar = data_plot.plot(ax=ax,add_colorbar=True,
                             vmin = cbar_range[0] if cbar_range else None,
                             vmax = cbar_range[1] if cbar_range else None)
        # Access and customize the colorbar
        if cbar.colorbar:
            cbar = cbar.colorbar
            cbar.ax.tick_params(labelsize=16)  # Adjust colorbar tick size
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(16)  # Adjust colorbar tick labels' font size
                
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)
        
    # Hide any unused subplots
    for j in range(n_data, len(axes)):
        axes[j].axis('off')     
    plt.tight_layout()
    plt.show()


def plot3d(datastore_tot, MOM_names_tot: list, var: str, datastore_ctrl = None, depth_level = None, common_time = None,
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10)):
    if not isinstance(datastore_tot,list):
        datastore_tot = [datastore_tot]
        
    n_data = len(datastore_tot)

    if nrows is None:
        nrows = (n_data+ncols-1)//ncols

    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)

    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    if datastore_ctrl is not None:
        cat_subset_contrl = datastore_ctrl.search(variable=[var])
        dset_contrl = cat_subset_contrl.to_dataset_dict(progressbar=False)
        key_contrl = list(dset_contrl.keys())[0]
        dset_sub_contrl = dset_contrl[key_contrl][var]

        if common_time is not None:
            tmp_contrl = dset_sub_contrl.sel(time=common_time,method='nearest')
        else:
            if time_selection == 'index':
                tmp_contrl = dset_sub_contrl.isel(time=time_index)
            elif time_selection == 'date':
                tmp_contrl = dset_sub_contrl.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp_contrl = dset_sub_contrl.sel(time=slice(start_time,end_time))
                tmp_contrl = tmp_contrl

    for _subplot, datastore in enumerate(datastore_tot):
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)

        key = list(dset.keys())[0]
        dset_sub = dset[key][var]

        if common_time is not None:
            tmp = dset_sub.sel(time=common_time,method='nearest')
        else:
            if time_selection == 'index':
                tmp = dset_sub.isel(time=time_index)
            elif time_selection == 'date':
                tmp = dset_sub.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp = dset_sub.sel(time=slice(start_time,end_time))

        if datastore_ctrl:
            data_plot = tmp - tmp_contrl
            title = f"diff - {MOM_names_tot[_subplot]}"
        else:
            data_plot = tmp
            title = f"{MOM_names_tot[_subplot]}"

        date_info = data_plot.time.values
        title += f"\n{date_info}"

        ax = axes[_subplot]

        if len(data_plot.dims) == 3:
            data_plot = data_plot.isel(zl=depth_level)
        
        cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                             vmin = cbar_range[0] if cbar_range else None,
                             vmax = cbar_range[1] if cbar_range else None,
                             cmap='RdBu')
        if cbar.colorbar:
            cbar = cbar.colorbar
            cbar.ax.tick_params(labelsize=16)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(16)

        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True)

    for j in range(n_data, len(axes)):
        axes[j].axis('off')     
    plt.tight_layout()
    plt.show()

