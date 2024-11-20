import os
import sys
import re
import copy
import subprocess
import shutil
import glob
import argparse
import warnings
from datetime import datetime, timedelta
try:
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import ScalarFormatter
    import git
    import f90nml
    import cftime
    import xarray as xr
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True

    import pandas as pd

    import pint_xarray
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

def read_ocean_stats(filepath, header=True):
    columns = [
        "Step", "Day", "Truncs", "Energy/Mass", "Maximum CFL", "Mean Sea Level",
        "Total Mass", "Mean Salin", "Mean Temp", "Frac Mass Err", "Salin Err", "Temp Err"
    ]
    stats = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith("  Step") or line.startswith("[days]"):
                continue
            tmp = line.split(",")
            if len(tmp)>1:
                step = int(tmp[0].strip())
                day = float(tmp[1].strip())
                truncs = int(tmp[2].strip())
                energy_mass = float(tmp[3].split()[1].strip())
                max_cfl = float(tmp[4].split()[1].strip())
                mean_sea_level = float(tmp[5].split()[1].strip())
                total_mass = float(tmp[6].split()[1].strip())
                mean_salin = float(tmp[7].split()[1].strip())
                mean_temp = float(tmp[8].split()[1].strip())
                frac_mass_err = float(tmp[9].split()[1].strip())
                salin_err = float(tmp[10].split()[1].strip())
                temp_err = float(tmp[11].split()[1].strip())
                stats.append(
                    [step, day, truncs, energy_mass, max_cfl, mean_sea_level, total_mass,
                             mean_salin, mean_temp, frac_mass_err, salin_err, temp_err]
                )
    if stats:
        df = pd.DataFrame(stats, columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    return df

def truncation_date(df_tmp, starting_date, dt):
    tmp_index = df_tmp[df_tmp['Truncs']!=0]['Step'].index
    year_index = (tmp_index//366).astype(int)
    new_starting_date = [starting_date.replace(year=starting_date.year+i) for i in year_index]
    tmp_values = df_tmp[df_tmp['Truncs']!=0]['Step'].values
    truncs_date = [output_dates(dt*tmp_values[i], new_starting_date[i]) for i in range(len(new_starting_date))]
    return truncs_date

def output_dates(input_secs, starting_date):
    date_time = starting_date + timedelta(seconds=int(input_secs))
    format_date = date_time.strftime('%Y-%m-%d %H:%M:%S')
    return format_date

def plot_time_series(datastore_tot, MOM_names, var, indices=None, base_index=None,
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
        # legend: supported values are 'best', 'upper right', 'upper left', 'lower left', 
        # 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
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
                print(i)
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
                                           markersize=markersize[i % len(markersize)],
                                           linewidth=line_width[i % len(line_width)],
                                           label=f"{MOM_names[i]} - Base" if j == 0 else "")
                            plot_ticks_labels()
                    else:
                        print(f"Base dataset for {key} not found.")
        else:
            for tmp_indx, i in enumerate(indices):
                print(i)
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
                                  markersize=markersize[i % len(markersize)],
                                  linewidth=line_width[i % len(line_width)],
                                  label=f"{MOM_names[i]}" if j == 0 else "")
                        plot_ticks_labels()
                    else:
                        pass

    if use_subplots:
        for j in range(n_data, len(axes)):
            axes[j].axis('off')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    # plt.tight_layout()
    # plt.show()
    # print(plt.get_fignums())

def plot2d(datastore_tot, MOM_names_tot: list, var: str, datastore_ctrl = None, depth_level = None, common_time = None,
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10), abs_threshold_value=None,
           xq_subset=None, yh_subset=None, cmap='RdBu_r', tick_fontsize=16, label_fontsize=16,
           cf_sum_lon=False, cf_sum_vertical=False, cf_mean_time=False, cf_mean_lon=False,
           yincrease=True):

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
        combined_data_contrl = combined_data_contrl.pint.quantify()

        if cf_sum_lon:
            combined_data_contrl = combined_data_contrl.cf.sum("longitude")
        if cf_mean_lon:
            combined_data_contrl = combined_data_contrl.cf.mean("longitude")
        if common_time is not None:
            tmp_contrl = combined_data_contrl.sel(time=common_time,method='nearest')
        else:
            if time_selection == 'index':
                tmp_contrl = combined_data_contrl.isel(time=time_index)
            elif time_selection == 'date':
                tmp_contrl = combined_data_contrl.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp_contrl = combined_data_contrl.sel(time=slice(start_time,end_time))

        if cf_sum_vertical:
            if cf_mean_time:
                combined_data_contrl = combined_data_contrl.cf.cumsum("vertical").mean("time") - combined_data_contrl.cf.sum("vertical").mean("time")
            else:
                combined_data_contrl = combined_data_contrl.cf.cumsum("vertical") - combined_data_contrl.cf.sum("vertical")

    for _subplot, datastore in enumerate(datastore_tot):
        print(_subplot)
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(dset)

        if common_time is not None:
            tmp = combined_data.sel(time=common_time, method='nearest')
            tmp = tmp.pint.quantify()
            if cf_sum_lon:
                tmp = tmp.cf.sum("longitude")
            if cf_mean_lon:
                tmp = tmp.cf.mean("longitude")
        else:
            if time_selection == 'index':
                tmp = combined_data.isel(time=time_index)
            elif time_selection == 'date':
                tmp = combined_data.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp = combined_data.sel(time=slice(start_time,end_time))
        if cf_sum_vertical:
            if cf_mean_time:
                tmp = tmp.cf.cumsum("vertical").mean("time") - tmp.cf.sum("vertical").mean("time")
            else:
                tmp = tmp.cf.cumsum("vertical") - tmp.cf.sum("vertical")

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
            if not cf_sum_vertical:
                data_plot = data_plot.isel(zl=depth_level)

        if abs_threshold_value is not None:
            mask_tmp = data_plot.where(abs(data_plot) > abs_threshold_value, np.nan)
            data_plot = mask_tmp
            xq_coords = data_plot['xq'].values
            yh_coords = data_plot['yh'].values
            print(xq_coords)
            print(yh_coords)
            mask = ~np.isnan(data_plot.values).flatten()
            non_nan_values = data_plot.values.flatten()[mask]
            non_nan_xq_coords = np.tile(xq_coords, (yh_coords.size, 1)).flatten()[mask]
            non_nan_yh_coords = np.repeat(yh_coords, xq_coords.size).flatten()[mask]
            non_nan_data = list(zip(non_nan_yh_coords, non_nan_xq_coords, non_nan_values))
            for yh, xq, val in non_nan_data:
                print(f"-- xq: {xq}, yh: {yh}, value: {val}")

        if (xq_subset and yh_subset) is not None:
            data_plot = data_plot.sel(xq=xq_subset, yh=yh_subset)

        if yincrease:
            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                 vmin = cbar_range[0] if cbar_range else None,
                                 vmax = cbar_range[1] if cbar_range else None,
                                 # cbar_kwargs={'label': data_plot.name},
                                 cmap = cmap)
        else:
            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                 vmin = cbar_range[0] if cbar_range else None,
                                 vmax = cbar_range[1] if cbar_range else None,
                                 # cbar_kwargs={'label': data_plot.name},
                                 cmap = cmap, yincrease=False)

        if cbar.colorbar:
            cbar = cbar.colorbar
            cbar.ax.tick_params(labelsize=16)
            for label in cbar.ax.get_yticklabels():
                label.set_fontsize(16)

            #cbar.ax.set_title(data_plot.name, fontsize=30, pad=10, loc='left')
            cbar.ax.yaxis.label.set_fontsize(30)

        ax.set_title(title)
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        #ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

    # ax = axes[3]
    # test_strings = "/g/data/tm70/ml0072/COMMON/git_repos/COSIMA_om3-scripts/expts_manager/product1_0.25deg/lexpt4_rr/archive/output000/access-om3.mom6.2d.T_adx_2d.1mon.snap.1900.nc"
    # test_xarray_dataset = xr.open_dataset(test_strings)
    # test_xarray_dataset = test_xarray_dataset.sel(time=common_time, method='nearest').pint.quantify()
    # print(test_xarray_dataset.time.values)
    # data_plot = test_xarray_dataset[var] - tmp_contrl[var]
    # cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
    #                              vmin = cbar_range[0] if cbar_range else None,
    #                              vmax = cbar_range[1] if cbar_range else None,
    #                              # cbar_kwargs={'label': data_plot.name},
    #                              cmap = cmap, yincrease=True)
    for j in range(n_data, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
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