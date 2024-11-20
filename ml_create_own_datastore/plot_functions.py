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
from collections import defaultdict
from pprint import pprint
try:
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.path as mpath
    import git
    import f90nml
    import cftime
    import xarray as xr
    from ruamel.yaml import YAML
    ryaml = YAML()
    ryaml.preserve_quotes = True

    import pandas as pd

    import pint_xarray
    from pint import application_registry as ureg
    import pint
    import cf_xarray.units
    from dask.distributed import Client
    import cartopy.crs as ccrs
    import cartopy.feature as cft
    #client = Client(threads_per_worker=1)
    #print(client)
    #print(client.scheduler_info())
    #print(client.ncores)

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

def combine_datasets(dset, keyword):
    # will be depreciated when https://github.com/ACCESS-NRI/access-nri-intake-catalog/pull/178#issuecomment-2415796731 is merged
    def get_time(dataset):
        if 'time' in dataset.coords:
            return dataset.time.values

    dataset_w_time = [(get_time(dataset), dataset) for key, dataset in dset.items() if keyword in key]
    darrays_sorted = [dataset for _, dataset in dataset_w_time]
    combined_data = xr.concat(darrays_sorted, dim='time')
    combined_data = combined_data.sortby('time')
    return combined_data

def sel_straits(strait):
    straits = {'Drake Passage': [ -69.9,  -69.9, -71.6, -51.0],
               'Lombok':        [-244.6, -243.9,  -8.6,  -8.6],
               'Ombai' :        [-235.0, -235.0,  -9.2,  -8.1],
               'Timor' :        [-235.9, -235.9, -11.9,  -9.9],
               'Denmark' :      [ -42.0,  -22.0,  65.8,  65.8],
               'Bering' :       [-172.0, -167.0,  65.8,  65.8],
               }
    xmin, xmax, ymin, ymax = straits[strait]
    return xmin, xmax, ymin, ymax

def trans_through_straits(trans, var, strait, xmin, xmax, ymin, ymax):
    rho0 = 1036*ureg.kilogram / ureg.meter**3

    if ymax >= 65:
        print("North of 65N the tripolar grid geometry brings complications and .sum(\'longitude\') is wrong!")
    print(f'Calculating {strait}')

    if xmin==xmax:
        if var == 'umo':
            transport = (trans/rho0).cf.sel(longitude=xmin, method="nearest").cf.sel(latitude=slice(ymin, ymax)).cf.sum({'vertical', 'latitude'})
            transport.attrs['long_name'] = 'zonal transport'
    elif ymin==ymax:
        if var == 'vmo':
            transport = (trans/rho0).cf.sel(longitude=slice(xmin, xmax)).cf.sel(latitude=ymin, method="nearest").cf.sum({'vertical', 'longitude'})
            transport.attrs['long_name'] = 'meridional transport'
    else:
            raise ValueError('Transports are computed only along lines of either constant latitude or constant longitude')

    transport = transport.pint.to('sverdrups')
    transport = transport.compute()

    return transport

# def load_topog_depth(topog_path = '/g/data/vk83/experiments/inputs/access-om3/share/grids/global.025deg/2023.05.15/topog.nc'):
#     topog = xr.open_dataset(topog_path)
#     depth = topog['depth']
#     return depth

def load_lons_lats(ocean_hgrid_path = '/g/data/tm70/ml0072/COMMON/git_repos/COSIMA_om3-scripts/expts_manager/product1_0.25deg/access-om3.mom6.static.nc'):
    ocean_hgrid = xr.open_dataset(ocean_hgrid_path)
    xh = ocean_hgrid.variables['xh']
    yh = ocean_hgrid.variables['yh']
    geolon = ocean_hgrid.variables['geolon']
    geolat = ocean_hgrid.variables['geolat']
    return xh, yh, geolon, geolat

# def load_deptho(ocean_hgrid_path = '/g/data/tm70/ml0072/COMMON/git_repos/COSIMA_om3-scripts/expts_manager/product1_0.25deg/access-om3.mom6.static.nc'):
#     ocean_hgrid = xr.open_dataset(ocean_hgrid_path)
#     deptho = ocean_hgrid.variables['deptho']
#     return deptho

def load_fx_dataset(datastore, var, frequency):
    tmp_xr_ds = datastore.search(variable=var, frequency=frequency, path=".*output000.*").to_dask()
    tmp_xr_da = tmp_xr_ds[var]
    return tmp_xr_da
    
def find_indices_based_on_xh_yh(target_lon, target_lat, xh, yh):
    if hasattr(xh, 'values'):
        xh = xh.values
    if hasattr(yh, 'values'):
        yh = yh.values

    lon_diff = np.abs(xh - target_lon)
    lat_diff = np.abs(yh - target_lat)

    nearest_lon_idx = np.argmin(lon_diff)
    nearest_lat_idx = np.argmin(lat_diff)
    nearest_idx = (nearest_lat_idx, nearest_lon_idx)

    nearest_lon = xh[nearest_lon_idx]
    nearest_lat = yh[nearest_lat_idx]
    nearest = (nearest_lat, nearest_lon)
    return nearest_idx, nearest_lon_idx, nearest_lat_idx, nearest_lon, nearest_lat, nearest

def convert_to_date_time(year, yearday, time):
    yearday = max(1, yearday)
    tmp = datetime(year, 1, 1)
    target_date = tmp + timedelta(days=yearday-1)

    hours = int(time)
    minutes = int((time-hours)*60)
    seconds = int((((time-hours)*60)-minutes)*60)
    target_datetime = target_date + timedelta(hours=hours, minutes=minutes, seconds=seconds)

    return target_datetime

def parse_truncation_file(truncation_file_path):
    xh, yh, geolon, geolat = load_lons_lats()
    trunc_pattern = re.compile(r'Time\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([UV])-velocity violation at\s+(\d+):\s+\d+\s+\d+\s+\(\s*([-+]?\d*\.\d+)\s+[E]?\s+([-+]?\d*\.\d+)\s+[N]?\)\s+Layers\s+(\d+)\s+to\s+(\d+)\.\s+dt\s+=\s+(\d+)')
    truncations = []
    if truncation_file_path:
        with open(truncation_file_path, 'r') as f:
            file_read = f.read()
        for match in trunc_pattern.finditer(file_read):
            year = int(match.group(1))
            yearday = int(match.group(2))
            time_of_day = float(match.group(3))
            velocity_type = match.group(4) + '-velocity'
            processor = int(match.group(5))
            lon = float(match.group(6))
            lat = float(match.group(7))
            layer_start = int(match.group(8))
            layer_end = int(match.group(9))
            dt = int(match.group(10))
    
            nearest_idx, nearest_lon_idx, nearest_lat_idx, nearest_lon, nearest_lat, nearest = find_indices_based_on_xh_yh(lon, lat, xh, yh)
            datetime_value = convert_to_date_time(year, yearday, time_of_day)
    
            truncations.append({
                'datetime': datetime_value,
                'velocity_type': velocity_type,
                'processor': processor,
                'longitude': lon,
                'latitude': lat,
                'longitude_index': nearest_lon_idx,
                'latitude_index': nearest_lat_idx,
                'eval_longitude': nearest_lon,
                'eval_latitude': nearest_lat,
                'layers_s': layer_start,
                'layers_e': layer_start,
                'dt': dt,
            })

    return truncations, xh, yh, geolon, geolat

def process_truncation_files(target_dir):
    all_truncation_data = []
    for output_dir in os.listdir(target_dir):
        output_dir_path = os.path.join(target_dir, output_dir)
        if os.path.isdir(output_dir_path) and output_dir.startswith('output'):
            for filename in os.listdir(output_dir_path):
                if filename.startswith("U_velocity_truncations") or filename.startswith("V_velocity_truncations"):
                    truncation_file_path = os.path.join(output_dir_path, filename)
                    truncation_data, xh, yh, geolon, geolat = parse_truncation_file(truncation_file_path)
                    if truncation_data:
                        all_truncation_data.append(truncation_data)
                else:
                    xh, yh, geolon, geolat = load_lons_lats()

    if not all_truncation_data:
        print(f'Warning: No velocity truncations exist in {target_dir}.')

    return all_truncation_data, xh, yh, geolon, geolat

def truncation_organise_by_variable(truncation_data, trunc_variable):
    group_data = defaultdict(list)
    for inner_list in truncation_data:
        for i in inner_list:
            if trunc_variable == 'trunc_month':
                v = i['datetime'].month
            elif trunc_variable == 'trunc_year':
                v = i['datetime'].year
            else:
                v = i[trunc_variable] if trunc_variable in i else None
            group_data[v].append(i)
    return dict(group_data)

def plot2d2(
        datastore_tot, MOM_names_tot: list, plots_config=None, data_collect_method=None,
        frequency_fx=None,
        common_time = None, time_selection = None, time_range=None,
        x_subset=None, y_subset=None,
        figsize=(15,10), ncols=2, nrows=None, cbar_range=None, cmap='RdBu_r', yincrease=True, levels=21,
        project_ccrs=False, project_ccrs_method=ccrs.PlateCarree(), geo_plot=False,
        tick_fontsize=12, label_fontsize=12,title_fontsize=12,
        depth_level = None,
        cf_sum_lon=False, cf_cumsum_vertical=False, cf_mean_time=False, cf_mean_lon=False,
        MOC=False, remap_vertical_to_depth_coord = False,
        Barotropic_streamfunction=False, ccrs_circumpolar_plot=False, land_mask_plot=False,
        truncation_errors=False, trunc_variable=None, trunc_plot=None, MOM_dirs_path_tot = False,
        plot_1d=False,
        line_style=None,
        line_color=None,
        line_marker=None,
        markersize=None,
        line_width=None,
        line_title=None,
        legend_loc='best',legend_fontsize=10,
        ):

    # load coords for data_plot
    xh, yh, geolon, geolat = load_lons_lats()

    # load land_50m
    land_50m = cft.NaturalEarthFeature(
        "physical", "land", "50m", edgecolor="black", facecolor="papayawhip", linewidth=0.5
    )

    def plot_time_ticks_labels(ylims):
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

    def convert_cftime_to_datetime(cftime_array):
        datetime_list = []
        for dt in cftime_array:
            dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                       hour=dt.hour, minute=dt.minute, second=dt.second)
            datetime_list.append(dt_datetime)
        return np.array(datetime_list)

    def datastore_to_combinedata(datastore, var):
        #print(datastore)
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(dset, data_collect_method)
        return combined_data

    def get_land_mask(datastore, var, frequency_fx):
        bathymetry_dict = load_fx_dataset(datastore, var, frequency_fx)
        #bathymetry_dict = load_deptho()
        #print(bathymetry_dict)
        land_mask = xr.where(np.isnan(bathymetry_dict), 1, np.nan)
        # land_mask.values
        # land_mask = xr.DataArray(land_mask, dims=['yh', 'xh'])
        land_mask = land_mask.rename('land_mask')
        return land_mask

    def circumpolar_map(ax):
        ax.set_extent([-180, 180, -80, -40], crs = ccrs.PlateCarree())
        ax.set_facecolor('lightgrey')
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform = ax.transAxes)
        return ax

    if not hasattr(ureg, 'psu'):  # pint does not have psu
        ureg.define('psu = []')

    if plots_config:
        indices = []
        for config in plots_config:
            indices.append(config.get('indices', list(range(len(datastore_tot)))))
    else:
        indices = [list(range(len(datastore_tot)))]
    print(f"all indices: {indices}")

    if plot_1d:
        num_plots = len(indices)
        print(f"total subplots: {num_plots} for 1d line plots")
    else:
        num_plots = sum([len(inner_list) for inner_list in indices])  # number of subplots
        print(f"total subplots: {num_plots} for 2d plots")

    if num_plots>1:
        if nrows is None:
            nrows = (num_plots + ncols - 1) // ncols
        if project_ccrs:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, subplot_kw={"projection": project_ccrs_method})
        else:
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if (nrows * ncols > 1) else [axes]
    else:
        if project_ccrs:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": project_ccrs_method})
        else:
            fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    if not plot_1d:
        plot_counter = 0

    for plot_indx, _subplot in enumerate(plots_config):
        title      = _subplot.get('title', '')
        base_index = _subplot.get('base_index', None)
        var        = _subplot.get('var', None)
        xlims      = _subplot.get('xlims', None)
        ylims      = _subplot.get('ylims', None)
        if plot_1d:
            line_style = ['-'] * len(indices[plot_indx]) if line_style is None else line_style
            line_color = ['k'] * len(indices[plot_indx]) if line_color is None else line_color
            line_marker = [''] * len(indices[plot_indx]) if line_marker is None else line_marker
            line_width = [5] * len(indices[plot_indx]) if line_width is None else line_width
            markersize = [0] * len(indices[plot_indx]) if markersize is None else markersize
            yh_select = _subplot.get('yh_select', None)
            select_yh = _subplot.get('select_yh', None)
            ax = axes[plot_indx]

        if base_index is not None:
            print(f"process base_index: {base_index}")
            combined_data_contrl = datastore_to_combinedata(datastore_tot[base_index], var)
            if common_time is not None:
                base_dict = combined_data_contrl.sel(time=common_time, method='nearest')
            else:
                if time_selection == 'date_slice':
                    base_dict = combined_data_contrl.sel(time=slice(time_range[0],time_range[1]))
                    
            base_dict = base_dict.pint.quantify()
            base_dict['time'] = convert_cftime_to_datetime(base_dict['time'].values)

            if cf_sum_lon:
                base_dict = base_dict.cf.sum("longitude")
            if cf_mean_lon:
                base_dict = base_dict.cf.mean("longitude")

            if MOC:
                # MOC needs to be associated with `cf_sum_lon`
                # MOC needs to be associated with `cf_cumsum_vertical`
                # MOC needs to be associated with `cf_mean_time`
                if var == 'vmo':
                    rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
                    base_dict = (base_dict/rho0).pint.to('sverdrup')
                else:
                    raise ValueError(f"variable {var} is not correct for MOC, it must be 'vmo' ")

            if Barotropic_streamfunction:
                # needs to be associated with cf_mean_time
                # needs to be associated with project_ccrs
                # needs to be associated with project_ccrs_method=ccrs.SouthPolarStereo()
                # needs to be associated with ccrs_circumpolar_plot
                # needs to be associated with land_mask_plot
                # needs to be associated with geo_plot
                if var == 'umo_2d':
                    rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
                    base_dict = (base_dict/rho0).pint.to('sverdrup')  # volume transport
                else:
                    raise ValueError(f"variable {var} is not correct for barotropic streamfunction, it must be 'umo_2d' ")
                yh_coords = base_dict["yh"]
                base_dict = base_dict.cf.cumsum('latitude')
                base_dict = base_dict.assign_coords(yh=yh_coords)
                #base_dict = base_dict.rename({var: 'psi'})
                base_dict.attrs['Standard name'] = 'Barotropic streamfunction'
                base_dict.attrs['units'] = base_dict[var].pint.units
                base_dict.pint.quantify()
            if land_mask_plot:
                land_mask = get_land_mask(datastore_tot[base_index], 'deptho', 'fx')

            if cf_cumsum_vertical:
                zl_coords = base_dict["zl"]
                base_dict_cumsum = base_dict.cf.cumsum("vertical")
                if cf_mean_time:
                    base_dict = base_dict_cumsum.mean("time") - base_dict.cf.sum("vertical").mean("time")
                else:
                    base_dict = base_dict_cumsum              - base_dict.cf.sum("vertical")
                base_dict = base_dict.assign_coords(zl=zl_coords)
            else:
                if cf_mean_time:
                    base_dict = base_dict.mean("time")

            print(f"finish processing base_index: {base_index}")

        for tmp_index,i in enumerate(_subplot['indices']):
            print(tmp_index,i)
            if not plot_1d:
                ax = axes[plot_counter]
            if frequency_fx == 'fx':
                data_plot = load_fx_dataset(datastore_tot[i], var, frequency_fx)
            else:
                combined_data = datastore_to_combinedata(datastore_tot[i], var)

                if common_time is not None:
                    dataset_dict = combined_data.sel(time=common_time, method='nearest')
                else:
                    if time_selection == 'date_slice':
                        dataset_dict = combined_data.sel(time=slice(time_range[0],time_range[1]))
    
                dataset_dict = dataset_dict.pint.quantify()
                dataset_dict['time'] = convert_cftime_to_datetime(dataset_dict['time'].values)

                if cf_sum_lon:
                    dataset_dict = dataset_dict.cf.sum("longitude")
                if cf_mean_lon:
                    dataset_dict = dataset_dict.cf.mean("longitude")

                if MOC:
                    # MOC needs to be associated with `cf_sum_lon`
                    # MOC needs to be associated with `cf_cumsum_vertical`
                    # MOC needs to be associated with `cf_mean_time`
                    rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
                    dataset_dict = (dataset_dict/rho0).pint.to('sverdrup')

                if Barotropic_streamfunction:
                    # needs to be associated with cf_mean_time
                    # needs to be associated with project_ccrs
                    # needs to be associated with project_ccrs_method=ccrs.SouthPolarStereo()
                    # needs to be associated with ccrs_circumpolar_plot
                    # needs to be associated with land_mask_plot
                    # needs to be associated with geo_plot
                    if var == 'umo_2d':
                        rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
                        dataset_dict = (dataset_dict/rho0).pint.to('sverdrup')  # volume transport
                    else:
                        raise ValueError(f"variable {var} is not correct for barotropic streamfunction, it must be 'umo_2d' ")
                    yh_coords = dataset_dict["yh"]
                    dataset_dict = dataset_dict.cf.cumsum('latitude')
                    dataset_dict = dataset_dict.assign_coords(yh=yh_coords)
                    #dataset_dict = dataset_dict.rename({var: 'psi'})
                    dataset_dict.attrs['Standard name'] = 'Barotropic streamfunction'
                    dataset_dict.attrs['units'] = dataset_dict[var].pint.units
                    dataset_dict.pint.quantify()
                    if base_index is None:
                        if land_mask_plot:
                            land_mask = get_land_mask(datastore_tot[i], 'deptho', 'fx')

                if cf_cumsum_vertical:
                    zl_coords = dataset_dict["zl"]
                    dataset_dict_cumsum = dataset_dict.cf.cumsum("vertical")
                    if cf_mean_time:
                        dataset_dict = dataset_dict_cumsum.mean("time") - dataset_dict.cf.sum("vertical").mean("time")
                    else:
                        dataset_dict = dataset_dict_cumsum              - dataset_dict.cf.sum("vertical")
                    dataset_dict = dataset_dict.assign_coords(zl=zl_coords)
                else:
                    if cf_mean_time:
                        dataset_dict = dataset_dict.mean("time")

                if common_time is not None:
                    if base_index:
                        if dataset_dict[var].time.values == base_dict[var].time.values:
                            data_plot = dataset_dict[var] - base_dict[var]
                            title = f"diff - {MOM_names_tot[i]}"
                        else:
                            raise ValueError(f"date is not consistent for the two datasets ",
                                  f"{dataset_dict[var].time.values} and {base_dict[var].time.values}"
                                 )
                    else:
                        data_plot = dataset_dict[var]
                        title = f"{MOM_names_tot[i]}"
                    date_info = data_plot.time.values
                    title += f"\n{date_info}"
                else:
                    if base_index is not None:
                        data_plot = dataset_dict[var] - base_dict[var]
                        title = f"diff - {MOM_names_tot[i]}"
                    else:
                        data_plot = dataset_dict[var]
                        title = f"{MOM_names_tot[i]}"
    
                if len(data_plot.dims) == 3:
                    if not cf_cumsum_vertical:
                        data_plot = data_plot.isel(zl=depth_level)
    
            if truncation_errors and MOM_dirs_path_tot:
                print('processing truncation error files...')
                truncation_data, xh, yh, _, _ = process_truncation_files(os.path.join(MOM_dirs_path_tot[i], 'archive'))
                #data_plot = load_topog_depth() # depth topography
                # data_plot = xr.DataArray(
                #     data=data_plot.values,
                #     dims=["yh", "xh"],
                #     coords={
                #         "yh": (["yh"], yh),
                #         "xh": (["xh"], xh),
                #     },
                #     attrs=data_plot.attrs
                # )
                group_data = truncation_organise_by_variable(truncation_data, trunc_variable)
                trunc_keys = list(group_data.keys())
                print(f"available trunc_keys: {trunc_keys}")
                title = f"{MOM_names_tot[i]}"

            if (x_subset and y_subset) is not None:
                coords = data_plot.coords
                if "xq" in coords and "yh" in coords:
                    data_plot = data_plot.sel(xq=x_subset, yh=y_subset)
                elif "xh" in coords and "yh" in coords:
                    data_plot = data_plot.sel(xh=x_subset, yh=y_subset)
                elif "xh" in coords and "yq" in coords:
                    data_plot = data_plot.sel(xh=x_subset, yq=y_subset)
                elif "xq" in coords and "yq" in coords:
                    data_plot = data_plot.sel(xq=x_subset, yq=y_subset)
                else:
                    raise ValueError("None of the expected coordinates (xq, xh, yq, yh) are found in data_plot.")

            if geo_plot:
                if not ccrs_circumpolar_plot:
                    if 'xh' in data_plot.coords or 'xq' in data_plot.coords:
                        lon_coord = 'xh' if 'xh' in data_plot.coords else 'xq'
                        data_plot = data_plot.assign_coords({lon_coord: geolon})
                    if 'yh' in data_plot.coords or 'yq' in data_plot.coords:
                        lat_coord = 'yh' if 'yh' in data_plot.coords else 'yq'
                        data_plot = data_plot.assign_coords({lat_coord: geolat})

            print('start plotting...')

            if plot_1d:
                if select_yh:
                    print(f'select_yh: {select_yh}')
                    data_plot = data_plot.sel(yh=yh_select,method='nearest')
                    nearest_yh = data_plot['yh'].values
                    line_title = f"{nearest_yh}\n[degree_north]"
                    # print(data_plot.coords)
                    #print(data_plot.values)
                    data_plot.plot(ax=ax,
                                   #x='zl',
                                   linestyle=line_style[tmp_index % len(line_style)],
                                   color=line_color[tmp_index % len(line_color)],
                                   marker=line_marker[tmp_index % len(line_marker)],
                                   markersize=markersize[tmp_index % len(markersize)],
                                   linewidth=line_width[tmp_index % len(line_width)],
                                   #label=f"{MOM_names_tot[tmp_index]}" # label=f"{MOM_names_tot[i]}"
                                  )
                    ax.set_title(line_title, fontsize=title_fontsize)
                    ax.legend(loc=legend_loc, fontsize=legend_fontsize)
            else:
                if yincrease:
                    if project_ccrs:
                        if geo_plot:
                            print('contourf21')
                            if ccrs_circumpolar_plot:
                                ax = circumpolar_map(ax)
                            cbar = data_plot.cf.plot.contourf(ax=ax,add_colorbar=True,
                                                 x='longitude',
                                                 y='latitude',
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 levels=levels,
                                                 transform=ccrs.PlateCarree(),
                                                 extend="both",
                                                 # cbar_kwargs={'label': data_plot.name},
                                                 cmap = cmap)
                            if land_mask_plot:
                                land_mask.plot.contourf(ax=ax,
                                                        colors = 'lightgrey',
                                                        add_colorbar = False, 
                                                        zorder = 2,
                                                        transform = ccrs.PlateCarree())
                        else:
                            print('pcolormesh21')
                            if ccrs_circumpolar_plot:
                                ax = circumpolar_map(ax)
                            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 transform=ccrs.PlateCarree(),
                                                 extend="both",
                                                 # cbar_kwargs={'label': data_plot.name},
                                                 cmap = cmap)
                    else:
                        if geo_plot:
                            print('contourf22')
                            cbar = data_plot.cf.plot.contourf(ax=ax,add_colorbar=True,
                                                 x='longitude',
                                                 y='latitude',
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 levels=levels,
                                                 extend="both",
                                                 cmap = cmap)
                        else:
                            print('pcolormesh22')
                            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 extend="both",
                                                 cmap = cmap)
                else:
                    if project_ccrs:
                        if geo_plot:
                            print('contourf21')
                            if ccrs_circumpolar_plot:
                                ax = circumpolar_map(ax)
                            cbar = data_plot.cf.plot.contourf(ax=ax,add_colorbar=True,
                                                 x='longitude',
                                                 y='latitude',
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 levels=levels,
                                                 transform=ccrs.PlateCarree(),
                                                 extend="both",
                                                 cmap = cmap, yincrease=False)
                        else:
                            print('pcolormesh21')
                            if ccrs_circumpolar_plot:
                                ax = circumpolar_map(ax)
                            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 extend="both",
                                                 cmap = cmap, yincrease=False)
                    else:
                        if geo_plot:
                            print('contourf22')
                            cbar = data_plot.cf.plot.contourf(ax=ax,add_colorbar=True,
                                                 x='longitude',
                                                 y='latitude',
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 levels=levels,
                                                 extend="both",
                                                 cmap = cmap, yincrease=False)
                        else:
                            print('pcolormesh22')
                            cbar = data_plot.plot.pcolormesh(ax=ax,add_colorbar=True,
                                                 vmin = cbar_range[0] if cbar_range else None,
                                                 vmax = cbar_range[1] if cbar_range else None,
                                                 extend="both",
                                                 cmap = cmap, yincrease=False)
                if project_ccrs:
                    if not ccrs_circumpolar_plot:
                        ax.coastlines(resolution="50m")
                        ax.add_feature(land_50m)
                    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linewidth=2, color='gray', alpha=0.5)
                    gl.xlabels_top = False
                    gl.ylabels_right = False
                    gl.xformatter = ccrs.cartopy.mpl.ticker.LongitudeFormatter()
                    gl.yformatter = ccrs.cartopy.mpl.ticker.LatitudeFormatter()
                    gl.xlabel_style = {'size': 15, 'color': 'black'}
                    gl.ylabel_style = {'size': 15, 'color': 'black'}
                    gl.xlocator = plt.MaxNLocator(5)
                    gl.ylocator = plt.MaxNLocator(5)
                    
                if cbar.colorbar:
                    cbar = cbar.colorbar
                    cbar.ax.tick_params(labelsize=tick_fontsize)
                    for label in cbar.ax.get_yticklabels():
                        label.set_fontsize(label_fontsize)
                    cbar.ax.yaxis.label.set_fontsize(label_fontsize)

                if truncation_errors and MOM_dirs_path_tot:
                    if trunc_plot is None:
                        trunc_plot = trunc_keys
    
                    if not isinstance(trunc_plot, list):
                        trunc_plot = [trunc_plot]

                    for trunc_tmp in trunc_plot:
                        if trunc_tmp in group_data:
                            print(trunc_tmp)
                            lon_indices = [item['longitude_index'] for item in group_data[trunc_tmp]]
                            lat_indices = [item['latitude_index'] for item in group_data[trunc_tmp]]
                            longitude = [item['longitude'] for item in group_data[trunc_tmp]]
                            latitude = [item['latitude'] for item in group_data[trunc_tmp]]
                            datetime = [item['datetime'] for item in group_data[trunc_tmp]]
                            pprint(f"lon_indices: {lon_indices}")
                            pprint(f"lat_indices: {lat_indices}")
                            pprint(f"datetime: {datetime}")
                            pprint(f"lon: {longitude}")
                            pprint(f"lat: {latitude}")
                            pprint(f"evaluate lon: {xh[lon_indices].values}")
                            pprint(f"evaluate lat: {yh[lat_indices].values}")

                            if project_ccrs:
                                ax.scatter(xh[lon_indices].values, yh[lat_indices].values, transform=ccrs.PlateCarree(), facecolors='none', edgecolors='r')
                            else:
                                ax.scatter(xh[lon_indices].values, yh[lat_indices].values, facecolors='none', edgecolors='r')

                ax.set_title(title, fontsize=title_fontsize)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
                ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
                ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

            if xlims:
                ax.set_xlim(xlims)
            else:
                ax.set_xlim(ax.get_xlim())
            if ylims is not None:
                ax.set_ylim(ylims)
            else:
                ax.set_ylim(ax.get_ylim())

            if not plot_1d:
                plot_counter += 1

    if num_plots>1:
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

def plot_time_series(datastore_tot, MOM_names, indices=None, base_index=None,
                     data_collect_method=None,
                    line_style=None, line_color=None, line_marker=None, markersize=None, line_width=None,
                    figsize=(15,10), tick_fontsize=12, label_fontsize=12, title_fontsize=12,
                    legend_loc='best',legend_fontsize=12,
                    plots_config=None, nrows=None, ncols=2,
                    time_range=None,
                    xlims=None, num_intervals=None, x_axis_time=True,
                    file_path = None,
                    ):

    def plot_time_ticks_labels(ylims):
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

    def convert_cftime_to_datetime(cftime_array):
        datetime_list = []
        for dt in cftime_array:
            dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
                                       hour=dt.hour, minute=dt.minute, second=dt.second)
            datetime_list.append(dt_datetime)
        return np.array(datetime_list)

    def datastore_to_dict(datastore_data, var, time_range):
        datastore_tmp = datastore_data.search(variable=var)
        #print(datastore_tmp.keys())
        datastore_tmp = datastore_tmp.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(datastore_tmp, data_collect_method)
        dataset_dict = combined_data[var].sel(time=slice(time_range[0], time_range[1]))
        return dataset_dict

    if not hasattr(ureg, 'psu'):  # pint does not have psu
        ureg.define('psu = []')

    if plots_config:
        indices = []
        for config in plots_config:
            indices.append(config.get('indices', list(range(len(datastore_tot)))))
    else:
        indices = [list(range(len(datastore_tot)))]

    num_plots = len(plots_config)  # number of subplots
    if num_plots>1:
        if nrows is None:
            nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if (nrows * ncols > 1) else [axes]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]

    # line_style = ['-'] * len(indices) if line_style is None else line_style
    # line_color = ['k'] * len(indices) if line_color is None else line_color
    # line_marker = [''] * len(indices) if line_marker is None else line_marker
    # line_width = [5] * len(indices) if line_width is None else line_width
    # markersize = [0] * len(indices) if markersize is None else markersize

    for plot_indx, _subplot in enumerate(plots_config):
        line_style = ['-'] * len(indices[plot_indx]) if line_style is None else line_style
        line_color = ['k'] * len(indices[plot_indx]) if line_color is None else line_color
        line_marker = [''] * len(indices[plot_indx]) if line_marker is None else line_marker
        line_width = [5] * len(indices[plot_indx]) if line_width is None else line_width
        markersize = [0] * len(indices[plot_indx]) if markersize is None else markersize
    
        title      = _subplot.get('title', '')
        base_index = _subplot.get('base_index', None)
        var        = _subplot.get('var', None)
        ylims      = _subplot.get('ylims', None)
        strait     = _subplot.get('strait', None)
        MHT_1      = _subplot.get('MHT_1', None)
        MHT_2      = _subplot.get('MHT_2', None)
        MHT_3      = _subplot.get('MHT_3', None)

        if num_plots>1:
            ax = axes[plot_indx]
        if base_index is not None:
            if MHT_3:
                var1 = 'thetao'
                var2 = 'vmo'
                base_dict_theta = datastore_to_dict(datastore_tot[base_index], var1, time_range)
                base_dict_theta = (base_dict_theta-273.15).assign_attrs(units='degree C')
                base_dict_V = datastore_to_dict(datastore_tot[base_index], var2, time_range)
                base_dict_theta = base_dict_theta.interp(yh=base_dict_V.yq.values, method='linear').rename({'yh': 'yq'})
                Cp = 3992.10322329649 ## heat capacity in J / (kg C); value used by MOM5
                base_data = (Cp * base_dict_V * base_dict_theta).cf.sum('longitude').cf.sum('vertical').mean('time')
                base_data = (base_data * 1e-15).assign_attrs(units='PettaWatts')
            else:
                base_dict = datastore_to_dict(datastore_tot[base_index], var, time_range)
                base_dict = base_dict.pint.quantify()
                base_dict['time'] = convert_cftime_to_datetime(base_dict['time'].values)

            if strait is not None:
                xmin, xmax, ymin, ymax = sel_straits(strait)
                base_data = trans_through_straits(base_dict, var, strait, xmin, xmax, ymin, ymax)

            if MHT_1:
                if var == "T_ady_2d":
                    base_dict = base_dict.pint.to('petawatt')
                    base_data = base_dict.mean('time').cf.sum('longitude')
                else:
                    raise ValueError("[var] has to be T_adx_2d for MHT 1st method!")

        for tmp_indx, i in enumerate(indices[plot_indx]):
            print(i)
            if MHT_3:
                var1 = 'thetao'
                var2 = 'vmo'
                dataset_dict_theta = datastore_to_dict(datastore_tot[i], var1, time_range)
                dataset_dict_theta = (dataset_dict_theta-273.15).assign_attrs(units='degree C')
                dataset_dict_V = datastore_to_dict(datastore_tot[i], var2, time_range)
                dataset_dict_theta = dataset_dict_theta.interp(yh=dataset_dict_V.yq.values, method='linear').rename({'yh': 'yq'})
                Cp = 3992.10322329649 ## heat capacity in J / (kg C); value used by MOM5
                data = (Cp * dataset_dict_V * dataset_dict_theta).cf.sum('longitude').cf.sum('vertical').mean('time')
                data = (data * 1e-15).assign_attrs(units='PettaWatts')
            else:
                dataset_dict = datastore_to_dict(datastore_tot[i], var, time_range)
                dataset_dict = dataset_dict.pint.quantify()
                dataset_dict['time'] = convert_cftime_to_datetime(dataset_dict['time'].values)
                data = dataset_dict

            if strait is not None:
                xmin, xmax, ymin, ymax = sel_straits(strait)
                data = trans_through_straits(dataset_dict, var, strait, xmin, xmax, ymin, ymax)

            if MHT_1:
                if var == "T_ady_2d":
                    #print(dataset_dict.attrs)
                    #print(dataset_dict)
                    dataset_dict = dataset_dict.pint.to('petawatt')
                    data = dataset_dict.mean('time').cf.sum('longitude')
                else:
                    raise ValueError("[var] has to be T_adx_2d for MHT 1st method!")

            if MHT_2:
                if var == "hfds":
                    dataset_dict = dataset_dict.pint.to('petawatt')
                    data = dataset_dict.mean('time')

                    area_dict = datastore_to_dict(datastore_tot[i], 'areacello', time_range)
                    lat_dict = datastore_to_dict(datastore_tot[i], 'geolat', time_range)
                    latv_dict = datastore_to_dict(datastore_tot[i], 'yh', time_range)
                    #TODO
                else:
                    raise ValueError("[var] has to be hfds for MHT 2nd method!")

            if base_index is not None:
                data_plot = data - base_data
                data_plot.plot(ax=ax,
                               linestyle=line_style[i % len(line_style)],
                               color=line_color[i % len(line_color)],
                               marker=line_marker[i % len(line_marker)],
                               markersize=markersize[i % len(markersize)],
                               linewidth=line_width[i % len(line_width)],
                               label=f"{MOM_names[i]} - Base")
            else:
                data_plot = data
                data_plot.plot(ax=ax,
                               linestyle=line_style[i % len(line_style)],
                               color=line_color[i % len(line_color)],
                               marker=line_marker[i % len(line_marker)],
                               markersize=markersize[i % len(markersize)],
                               linewidth=line_width[i % len(line_width)],
                               label=f"{MOM_names[i]}")
        if x_axis_time:
            plot_time_ticks_labels(ylims)
        else:
            if xlims:
                ax.set_xlim(xlims)
            else:
                ax.set_xlim(ax.get_xlim())

        ax.set_title(title, fontsize=title_fontsize)
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)

        ax.grid(color='gray', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

    if num_plots>1:
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

# def plot_time_series(datastore_tot, MOM_names, var, indices=None, base_index=None,
#                      data_collect_method=None,
#                     line_style=None, line_color=None, line_marker=None, markersize=None, line_width=None,
#                     figsize=(15,10), tick_fontsize=12, label_fontsize=12, title_fontsize=12,
#                     legend_loc='best',legend_fontsize=12,
#                     plots_config=None, use_subplots=False, nrows=None, ncols=2,
#                     time_range=None,
#                     xlims=None, num_intervals=None, ylims=None, x_axis_time=True,
#                     file_path = None, strait=None, MHT_1=False, MHT_2=False, MHT_3=False,
#                     ):

#     def plot_time_ticks_labels():
#         if ylims is not None:
#             ax.set_ylim(ylims)
#         if xlims is not None:
#             xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
#             ax.set_xlim(xlims_pd)
#             if num_intervals is not None:
#                 tick_positions = np.linspace(start=xlims_pd[0].value, stop=xlims_pd[1].value, num=num_intervals + 2)
#                 tick_positions = pd.to_datetime(tick_positions)
#                 ax.set_xticks(tick_positions)
#                 custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
#                 ax.set_xticklabels(custom_labels)

#     def convert_cftime_to_datetime(cftime_array):
#         datetime_list = []
#         for dt in cftime_array:
#             dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
#                                        hour=dt.hour, minute=dt.minute, second=dt.second)
#             datetime_list.append(dt_datetime)
#         return np.array(datetime_list)

#     def datastore_to_dict(datastore_data, var, time_range):
#         datastore_tmp = datastore_data.search(variable=var)
#         print(datastore_tmp.keys())
#         datastore_tmp = datastore_tmp.to_dataset_dict(progressbar=False)
#         combined_data = combine_datasets(datastore_tmp, data_collect_method)
#         dataset_dict = combined_data[var].sel(time=slice(time_range[0], time_range[1]))

#         return dataset_dict

#     if plots_config:
#         indices = plots_config[0].get('indices', list(range(len(datastore_tot))))
#     else:
#         indices = list(range(len(datastore_tot)))

#     if use_subplots:
#         n_data = len(indices)
#         if nrows is None:
#             nrows = (n_data + ncols - 1) // ncols
#         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#         axes = axes.flatten() if (nrows * ncols > 1) else [axes]
#     else:
#         fig, ax = plt.subplots(figsize=figsize)
#         axes = [ax]

#     line_style = ['-'] * len(indices) if line_style is None else line_style
#     line_color = ['k'] * len(indices) if line_color is None else line_color
#     line_marker = [''] * len(indices) if line_marker is None else line_marker
#     line_width = [5] * len(indices) if line_width is None else line_width
#     markersize = [0] * len(indices) if markersize is None else markersize

#     for plot_indx, _subplot in enumerate(plots_config):
#         title = _subplot.get('title', '')
#         base_index = _subplot.get('base_index', None)
#         if base_index is not None:

#             if MHT_3:
#                 var1 = 'thetao'
#                 var2 = 'vmo'
#                 base_dict_theta = datastore_to_dict(datastore_tot[base_index], var1, time_range)
#                 base_dict_theta = (base_dict_theta-273.15).assign_attrs(units='degree C')
#                 base_dict_V = datastore_to_dict(datastore_tot[base_index], var2, time_range)
#                 base_dict_theta = base_dict_theta.interp(yh=base_dict_V.yq.values, method='linear').rename({'yh': 'yq'})
#                 Cp = 3992.10322329649 ## heat capacity in J / (kg C); value used by MOM5
#                 base_data = (Cp * base_dict_V * base_dict_theta).cf.sum('longitude').cf.sum('vertical').mean('time')
#                 base_data = (base_data * 1e-15).assign_attrs(units='PettaWatts')
#             else:
#                 base_dict = datastore_to_dict(datastore_tot[base_index], var, time_range)
#                 base_dict = base_dict.pint.quantify()
#                 base_dict['time'] = convert_cftime_to_datetime(base_dict['time'].values)

#             if strait is not None:
#                 xmin, xmax, ymin, ymax = sel_straits(strait)
#                 base_data = trans_through_straits(base_dict, var, strait, xmin, xmax, ymin, ymax)

#             if MHT_1:
#                 if var == "T_ady_2d":
#                     base_dict = base_dict.pint.to('petawatt')
#                     base_data = base_dict.mean('time').cf.sum('longitude')
#                 else:
#                     raise ValueError("[var] has to be T_adx_2d for MHT 1st method!")

#         for tmp_indx, i in enumerate(indices):
#             print(i)
#             if use_subplots:
#                 ax = axes[tmp_indx]

#             if MHT_3:
#                 var1 = 'thetao'
#                 var2 = 'vmo'
#                 dataset_dict_theta = datastore_to_dict(datastore_tot[i], var1, time_range)
#                 dataset_dict_theta = (dataset_dict_theta-273.15).assign_attrs(units='degree C')
#                 dataset_dict_V = datastore_to_dict(datastore_tot[i], var2, time_range)
#                 dataset_dict_theta = dataset_dict_theta.interp(yh=dataset_dict_V.yq.values, method='linear').rename({'yh': 'yq'})
#                 Cp = 3992.10322329649 ## heat capacity in J / (kg C); value used by MOM5
#                 data = (Cp * dataset_dict_V * dataset_dict_theta).cf.sum('longitude').cf.sum('vertical').mean('time')
#                 data = (data * 1e-15).assign_attrs(units='PettaWatts')
#             else:
#                 dataset_dict = datastore_to_dict(datastore_tot[i], var, time_range)
#                 dataset_dict = dataset_dict.pint.quantify()
#                 dataset_dict['time'] = convert_cftime_to_datetime(dataset_dict['time'].values)
#                 data = dataset_dict

#             if strait is not None:
#                 xmin, xmax, ymin, ymax = sel_straits(strait)
#                 data = trans_through_straits(dataset_dict, var, strait, xmin, xmax, ymin, ymax)

#             if MHT_1:
#                 if var == "T_ady_2d":
#                     #print(dataset_dict.attrs)
#                     #print(dataset_dict)
#                     dataset_dict = dataset_dict.pint.to('petawatt')
#                     data = dataset_dict.mean('time').cf.sum('longitude')
#                 else:
#                     raise ValueError("[var] has to be T_adx_2d for MHT 1st method!")

#             if MHT_2:
#                 if var == "hfds":
#                     dataset_dict = dataset_dict.pint.to('petawatt')
#                     data = dataset_dict.mean('time')

#                     area_dict = datastore_to_dict(datastore_tot[i], 'areacello', time_range)
#                     lat_dict = datastore_to_dict(datastore_tot[i], 'geolat', time_range)
#                     latv_dict = datastore_to_dict(datastore_tot[i], 'yh', time_range)
#                     #TODO
#                 else:
#                     raise ValueError("[var] has to be hfds for MHT 2nd method!")

#             if base_index is not None:
#                 data_plot = data - base_data
#                 data_plot.plot(ax=ax,
#                                linestyle=line_style[i % len(line_style)],
#                                color=line_color[i % len(line_color)],
#                                marker=line_marker[i % len(line_marker)],
#                                markersize=markersize[i % len(markersize)],
#                                linewidth=line_width[i % len(line_width)],
#                                label=f"{MOM_names[i]} - Base")
#             else:
#                 data_plot = data
#                 data_plot.plot(ax=ax,
#                                linestyle=line_style[i % len(line_style)],
#                                color=line_color[i % len(line_color)],
#                                marker=line_marker[i % len(line_marker)],
#                                markersize=markersize[i % len(markersize)],
#                                linewidth=line_width[i % len(line_width)],
#                                label=f"{MOM_names[i]}")
#         if x_axis_time:
#             plot_time_ticks_labels()
#         else:
#             if xlims:
#                 ax.set_xlim(xlims)
#             else:
#                 ax.set_xlim(ax.get_xlim())

#     ax.set_title(title, fontsize=title_fontsize)
#     ax.legend(loc=legend_loc, fontsize=legend_fontsize)
#     ax.grid(color='gray', linestyle='--', linewidth=0.5)
#     ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
#     ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
#     ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

#     if use_subplots:
#         for j in range(n_data, len(axes)):
#             axes[j].axis('off')
#     plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    
# def plot_time_series(datastore_tot, MOM_names, var, indices=None, base_index=None,
#                     line_style=None, line_color=None, line_marker=None, markersize=None, line_width=None,
#                     figsize=(15,10), tick_fontsize=12, label_fontsize=12, title_fontsize=12,
#                     legend_loc='best',legend_fontsize=12,
#                     plots_config=None, use_subplots=False, nrows=None, ncols=2,
#                     time_range=None,
#                     xlims=None, num_intervals=None, ylims=None,
#                     file_path = None, strait=None):

#     def dict_data_and_sorted_keys(data_store, var):
#         datastore_tmp = data_store.search(variable=var)
#         dataset_dict = datastore_tmp.to_dataset_dict(progressbar=False)
#         sorted_keys = sorted(dataset_dict.keys(), key=lambda x: int(x.split('_')[-1].split('.')[0][-4:]))
#         return dataset_dict, sorted_keys

#     def plot_ticks_labels():
#         if ylims is not None:
#             ax.set_ylim(ylims)
#         if xlims is not None:
#             xlims_pd = (pd.to_datetime(xlims[0]), pd.to_datetime(xlims[1]))
#             ax.set_xlim(xlims_pd)
#             if num_intervals is not None:
#                 tick_positions = np.linspace(start=xlims_pd[0].value, stop=xlims_pd[1].value, num=num_intervals + 2)
#                 tick_positions = pd.to_datetime(tick_positions)
#                 ax.set_xticks(tick_positions)
#                 custom_labels = [f'{tick.year:04d}-{tick.month:02d}-{tick.day:02d}' for tick in tick_positions]
#                 ax.set_xticklabels(custom_labels)
#         ax.set_title(title, fontsize=title_fontsize)
#         ax.legend(loc=legend_loc, fontsize=legend_fontsize)
#         ax.grid(color='gray', linestyle='--', linewidth=0.5)
#         ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
#         ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
#         ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

#     def set_time_range(dataset, var, time_range):
#         if time_range is not None:
#             start,end = time_range
#             return dataset[var].sel(time=slice(start, end))
#         return dataset[var]

#     def convert_cftime_to_datetime(cftime_array):
#         datetime_list = []
#         for dt in cftime_array:
#             dt_datetime = pd.Timestamp(year=dt.year, month=dt.month, day=dt.day,
#                                        hour=dt.hour, minute=dt.minute, second=dt.second)
#             datetime_list.append(dt_datetime)
#         return np.array(datetime_list)
        
#     if plots_config:
#         indices = plots_config[0].get('indices', list(range(len(datastore_tot))))
#     else:
#         indices = list(range(len(datastore_tot)))

#     if use_subplots:
#         n_data = len(indices)
#         if nrows is None:
#             nrows = (n_data + ncols - 1) // ncols
#         fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#         axes = axes.flatten() if (nrows * ncols > 1) else [axes]
#     else:
#         fig, ax = plt.subplots(figsize=figsize)
#         axes = [ax]

#     line_style = ['-'] * len(indices) if line_style is None else line_style
#     line_color = ['k'] * len(indices) if line_color is None else line_color
#     line_marker = [''] * len(indices) if line_marker is None else line_marker
#     line_width = [5] * len(indices) if line_width is None else line_width
#     markersize = [0] * len(indices) if markersize is None else markersize

#     for plot_indx, _subplot in enumerate(plots_config):
#         title = _subplot.get('title', '')
#         base_index = _subplot.get('base_index', None)
#         if base_index is not None:
#             base_dict, sorted_base_keys = dict_data_and_sorted_keys(datastore_tot[base_index], var)

#             for tmp_indx, i in enumerate(indices):
#                 print(i)
#                 if use_subplots:
#                     ax = axes[tmp_indx]
#                 dataset_dict, sorted_keys = dict_data_and_sorted_keys(datastore_tot[i], var)

#                 for j, key in enumerate(sorted_keys):
#                     # expts datasets
#                     dataset = dataset_dict[key]
#                     data = set_time_range(dataset, var, time_range)
#                     data['time'] = convert_cftime_to_datetime(data['time'].values)
#                     data = data.pint.quantify()
#                     if strait is not None:
#                         xmin, xmax, ymin, ymax = sel_straits(strait)
#                         data = trans_through_straits(data, var, strait, xmin, xmax, ymin, ymax)

#                     # control dataset
#                     base_data = base_dict.get(sorted_base_keys[j], None)
#                     if base_data is not None:
#                         base_data_aligned = set_time_range(base_data, var, time_range)
#                         base_data_aligned['time'] = convert_cftime_to_datetime(base_data_aligned['time'].values)
#                         base_data_aligned = base_data_aligned.pint.quantify()
#                         if strait is not None:
#                             xmin, xmax, ymin, ymax = sel_straits(strait)
#                             data = trans_through_straits(base_data_aligned, var, strait, xmin, xmax, ymin, ymax)

#                         # calc difference
#                         data_diff = data - base_data_aligned

#                         # plot
#                         if not data_diff.isnull().all():
#                             data_diff.plot(ax=ax,
#                                            linestyle=line_style[i % len(line_style)],
#                                            color=line_color[i % len(line_color)],
#                                            marker=line_marker[i % len(line_marker)],
#                                            markersize=markersize[i % len(markersize)],
#                                            linewidth=line_width[i % len(line_width)],
#                                            label=f"{MOM_names[i]} - Base" if j == 0 else "")
#                             plot_ticks_labels()
#                     else:
#                         print(f"Base dataset for {key} not found.")
#         else:
#             for tmp_indx, i in enumerate(indices):
#                 print(i)
#                 if use_subplots:
#                     ax = axes[tmp_indx]
#                 dataset_dict, sorted_keys = dict_data_and_sorted_keys(datastore_tot[i], var)

#                 for j, key in enumerate(sorted_keys):
#                     dataset = dataset_dict[key]
#                     data = set_time_range(dataset, var, time_range)
#                     data['time'] = convert_cftime_to_datetime(data['time'].values)
#                     data = data.pint.quantify()

#                     if strait is not None:
#                         xmin, xmax, ymin, ymax = sel_straits(strait)
#                         data = trans_through_straits(data, var, strait, xmin, xmax, ymin, ymax)

#                     if not data.isnull().all():
#                         data.plot(ax=ax,
#                                   linestyle=line_style[i % len(line_style)],
#                                   color=line_color[i % len(line_color)],
#                                   marker=line_marker[i % len(line_marker)],
#                                   markersize=markersize[i % len(markersize)],
#                                   linewidth=line_width[i % len(line_width)],
#                                   label=f"{MOM_names[i]}" if j == 0 else "")
#                         plot_ticks_labels()
#                     else:
#                         pass

#     if use_subplots:
#         for j in range(n_data, len(axes)):
#             axes[j].axis('off')
#     plt.gca().yaxis.set_major_formatter(ScalarFormatter())

def plot2d(datastore_tot, MOM_names_tot: list, var: str, data_collect_method=None,
           plot_1d=False,yh_select=None,
           line_style=None, line_color=None, line_marker=None, markersize=None, line_width=None,legend_loc='best',legend_fontsize=10,
           line_title=None,
           datastore_ctrl = None, depth_level = None, common_time = None,
           time_selection = None, compute_stats = None, time_index = None, start_time=None, end_time=None, 
           ncols=2, nrows=None,cbar_range=None, figsize=(15,10),
           x_subset=None, y_subset=None, cmap='RdBu_r', tick_fontsize=12, label_fontsize=12,title_fontsize=12,
           cf_sum_lon=False, cf_cumsum_vertical=False, cf_mean_time=False, cf_mean_lon=False,
           yincrease=True, levels=21,
           MOC=False,
           remap_vertical_to_depth_coord = False,
           truncation_errors=False, trunc_variable=None, trunc_plot=None, MOM_dirs_path_tot = False,
           xlims=None, ylims=None,
          ):

    def datastore_to_combinedata(datastore, var):
        cat_subset = datastore.search(variable=[var])
        dset = cat_subset.to_dataset_dict(progressbar=False)
        combined_data = combine_datasets(dset, data_collect_method)
        return combined_data

    if not isinstance(datastore_tot,list):
        datastore_tot = [datastore_tot]
    n_data = len(datastore_tot)

    if plot_1d:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        if nrows is None:
            nrows = (n_data + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if (nrows * ncols > 1) else [axes]

    line_style = ['-'] * n_data if line_style is None else line_style
    line_color = ['k'] * n_data if line_color is None else line_color
    line_marker = [''] * n_data if line_marker is None else line_marker
    line_width = [5] * n_data if line_width is None else line_width
    markersize = [0] * n_data if markersize is None else markersize

    # if plot_1d:
    #     nrows=1
    #     ncols=1
    #     fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
    #     axes = axes.flatten() if (nrows * ncols > 1) else [axes]
    # else:
    #     if nrows is None:
    #         nrows = (n_data+ncols-1)//ncols
    #     fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=figsize)
    #     axes = axes.flatten() if (nrows * ncols > 1) else [axes]

    if datastore_ctrl is not None:
        combined_data_contrl = datastore_to_combinedata(datastore_ctrl, var)

        if common_time is not None:
            tmp_contrl = combined_data_contrl.sel(time=common_time, method='nearest')
        else:
            if time_selection == 'index':
                tmp_contrl = combined_data_contrl.isel(time=time_index)
            elif time_selection == 'date':
                tmp_contrl = combined_data_contrl.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp_contrl = combined_data_contrl.sel(time=slice(start_time,end_time))

        tmp_contrl = tmp_contrl.pint.quantify()

        if cf_sum_lon:
            tmp_contrl = tmp_contrl.cf.sum("longitude")
        if cf_mean_lon:
            tmp_contrl = tmp_contrl.cf.mean("longitude")

        if MOC:
            # MOC needs to be associated with `cf_sum_lon`
            # MOC needs to be associated with `cf_cumsum_vertical`
            # MOC needs to be associated with `cf_mean_time`
            rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
            tmp_contrl = (tmp_contrl/rho0).pint.to('sverdrup')

        if cf_cumsum_vertical:
            if cf_mean_time:
                tmp_contrl = tmp_contrl.cf.cumsum("vertical").mean("time") - tmp_contrl.cf.sum("vertical").mean("time")
            else:
                tmp_contrl = tmp_contrl.cf.cumsum("vertical") - tmp_contrl.cf.sum("vertical")
        else:
            if cf_mean_time:
                tmp_contrl = tmp_contrl.mean("time")

    for _subplot, datastore in enumerate(datastore_tot):
        print(_subplot)
        combined_data = datastore_to_combinedata(datastore, var)
        # print(combined_data.attrs)
        # print(combined_data)
        if common_time is not None:
            tmp = combined_data.sel(time=common_time, method='nearest')
        else:
            if time_selection == 'index':
                tmp = combined_data.isel(time=time_index)
            elif time_selection == 'date':
                tmp = combined_data.sel(time=start_time)
            elif time_selection == 'date_slice':
                tmp = combined_data.sel(time=slice(start_time,end_time))

        tmp = tmp.pint.quantify()

        if cf_sum_lon:
            tmp = tmp.cf.sum("longitude")
        if cf_mean_lon:
            tmp = tmp.cf.mean("longitude")

        if MOC:
            # MOC needs to be associated with `cf_sum_lon`
            # MOC needs to be associated with `cf_cumsum_vertical`
            # MOC needs to be associated with `cf_mean_time`
            rho0 = 1025 * ureg.kilogram / ureg.meter**3 # mean density of sea-water
            tmp = (tmp/rho0).pint.to('sverdrup')

        if remap_vertical_to_depth_coord:
            rho2 = datastore_to_combinedata(datastore, 'rhopot2')

            rho2 = rho2.pint.quantify()

            # mask the Mediteranean
            rho2 = rho2.cf.where(((rho2.cf['longitude'] < 0) | (rho2.cf['longitude'] > 45) ) |
                                 ((rho2.cf['latitude'] < 10) | (rho2.cf['latitude'] > 48))
                                )
            rho2_time_mean = rho2.sel(time=slice(start_time, end_time)).mean('time')
            rho2_zonal_mean = rho2_time_mean.cf.mean("longitude")

            # nmin is the latitude index that corresponds to 78S
            #TODO

        if cf_cumsum_vertical:
            if cf_mean_time:
                tmp = tmp.cf.cumsum("vertical").mean("time") - tmp.cf.sum("vertical").mean("time")
            else:
                tmp = tmp.cf.cumsum("vertical") - tmp.cf.sum("vertical")
        else:
            if cf_mean_time:
                tmp = tmp.mean("time")
                
        if truncation_errors and MOM_dirs_path_tot:
            print('processing truncation error files...')
            data_plot = load_topog_depth() # depth topography
            truncation_data = process_truncation_files(MOM_dirs_path_tot[_subplot])
            group_data = truncation_organise_by_variable(truncation_data, trunc_variable)
            trunc_keys = list(group_data.keys())
            print(f"available trunc_keys: {trunc_keys}")

        if common_time is not None:
            if datastore_ctrl:
                if tmp[var].time.values == tmp_contrl[var].time.values:
                    data_plot = tmp[var] - tmp_contrl[var]
                    title = f"diff - {MOM_names_tot[_subplot]}"
                else:
                    raise ValueError(f"date is not consistent for the two datasets ",
                          f"{tmp[var].time.values} and {tmp_contrl[var].time.values}"
                         )
            else:
                data_plot = tmp[var]
                title = f"{MOM_names_tot[_subplot]}"
            date_info = data_plot.time.values
            title += f"\n{date_info}"
        elif truncation_errors:
            title = f"{MOM_names_tot[_subplot]}"
        else:
            if datastore_ctrl:
                data_plot = tmp[var] - tmp_contrl[var]
                title = f"diff - {MOM_names_tot[_subplot]}"
            else:
                data_plot = tmp[var]
                title = f"{MOM_names_tot[_subplot]}"
        if plot_1d:
            ax=axes[0]
        else:
            ax = axes[_subplot]

        if len(data_plot.dims) == 3:
            if not cf_cumsum_vertical:
                data_plot = data_plot.isel(zl=depth_level)

        if (x_subset and y_subset) is not None:
            data_plot = data_plot.sel(xq=x_subset, yh=y_subset)

        if plot_1d:
            data_plot = data_plot.sel(yh=yh_select,method='nearest')
            nearest_yh = data_plot['yh'].values
            line_title = f"{nearest_yh}\n[degree_north]"
            data_plot.plot(ax=ax,
                           linestyle=line_style[_subplot % len(line_style)],
                           color=line_color[_subplot % len(line_color)],
                           marker=line_marker[_subplot % len(line_marker)],
                           markersize=markersize[_subplot % len(markersize)],
                           linewidth=line_width[_subplot % len(line_width)],
                           label=f"{MOM_names_tot[_subplot]}"
                          )
            ax.set_title(line_title, fontsize=title_fontsize)
            ax.legend(loc=legend_loc, fontsize=legend_fontsize)
        else:
            if yincrease:
                cbar = data_plot.plot.contourf(ax=ax,add_colorbar=True,
                                     vmin = cbar_range[0] if cbar_range else None,
                                     vmax = cbar_range[1] if cbar_range else None,
                                     levels=levels,
                                     # cbar_kwargs={'label': data_plot.name},
                                     cmap = cmap)
            else:
                cbar = data_plot.plot.contourf(ax=ax,add_colorbar=True,
                                     vmin = cbar_range[0] if cbar_range else None,
                                     vmax = cbar_range[1] if cbar_range else None,
                                     levels=levels,
                                     # cbar_kwargs={'label': data_plot.name},
                                     cmap = cmap, yincrease=False)
            if cbar.colorbar:
                cbar = cbar.colorbar
                cbar.ax.tick_params(labelsize=tick_fontsize)
                for label in cbar.ax.get_yticklabels():
                    label.set_fontsize(label_fontsize)
                cbar.ax.yaxis.label.set_fontsize(label_fontsize)
    
            if truncation_errors and MOM_dirs_path_tot:
                if trunc_plot is None:
                    trunc_plot = trunc_keys
    
                if not isinstance(trunc_plot, list):
                    trunc_plot = [trunc_plot]
    
                for trunc_tmp in trunc_plot:
                    if trunc_tmp in group_data:
                        print(trunc_tmp)
                        lon_indices = [item['longitude_index'] for item in group_data[trunc_tmp]]
                        lat_indices = [item['latitude_index'] for item in group_data[trunc_tmp]]
                        longitude = [item['longitude'] for item in group_data[trunc_tmp]]
                        latitude = [item['latitude'] for item in group_data[trunc_tmp]]
                        datetime = [item['datetime'] for item in group_data[trunc_tmp]]
                        pprint(f"lon_indices: {lon_indices}")
                        pprint(f"lat_indices: {lat_indices}")
                        pprint(f"datetime: {datetime}")
                        pprint(f"lon: {longitude}")
                        pprint(f"lat: {latitude}")
                        plt.scatter(lon_indices, lat_indices, facecolors='none', edgecolors='r')

            ax.set_title(title, fontsize=title_fontsize)
            ax.grid(True)
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)

        if xlims:
            ax.set_xlim(xlims)
        else:
            ax.set_xlim(ax.get_xlim())
        if ylims is not None:
            ax.set_ylim(ylims)

    for j in range(n_data, len(axes)):
        axes[j].axis('off')



# client.close()
