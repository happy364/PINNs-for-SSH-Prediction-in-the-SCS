import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt


def process_era5_wind_data_with_normalization(input_dir, output_dir, start_year=1993, end_year=2025):
    """
    批量处理ERA5风速数据，裁剪区域并插值到1/8度分辨率，添加标准化处理

    Parameters:
    input_dir (str): 输入数据目录
    output_dir (str): 输出数据目录
    start_year (int): 起始年份
    end_year (int): 结束年份
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 用于收集所有年份的数据以计算统计量
    all_u10_data = []
    all_v10_data = []

    # 处理每一年的数据
    for year in range(start_year, end_year + 1):
        print(f"正在处理 {year} 年数据...")

        path_dir1 = f'era5_daily_mean_u10_v10_{year}_1'
        path_dir2 = f'era5_daily_mean_u10_v10_{year}_2'

        # 构造文件路径
        file_pattern_1_u = os.path.join(input_dir, path_dir1,'10m_u_component_of_wind_stream-oper_daily-mean.nc')
        file_pattern_1_v = os.path.join(input_dir, path_dir1,'10m_v_component_of_wind_0_daily-mean.nc')
        file_pattern_2_u = os.path.join(input_dir, path_dir2,'10m_u_component_of_wind_stream-oper_daily-mean.nc')
        file_pattern_2_v = os.path.join(input_dir, path_dir2,'10m_v_component_of_wind_0_daily-mean.nc')


        # 加载数据
        ds1u = xr.open_dataset(file_pattern_1_u).astype(np.float32)
        ds1v = xr.open_dataset(file_pattern_1_v).astype(np.float32)
        ds2u = xr.open_dataset(file_pattern_2_u).astype(np.float32)
        ds2v = xr.open_dataset(file_pattern_2_v).astype(np.float32)
        print("load succeed")

        # 在合并之前重命名维度
        ds1u = ds1u.rename({'valid_time': 'time'}) if 'valid_time' in ds1u.dims else ds1u
        ds1v = ds1v.rename({'valid_time': 'time'}) if 'valid_time' in ds1v.dims else ds1v
        ds2u = ds2u.rename({'valid_time': 'time'}) if 'valid_time' in ds2u.dims else ds2u
        ds2v = ds2v.rename({'valid_time': 'time'}) if 'valid_time' in ds2v.dims else ds2v

        ds_u = xr.concat([ds1u, ds2u], dim='time')
        del ds1u
        del ds2u
        print('time concate')
        ds_v = xr.concat([ds1v, ds2v], dim='time')
        del ds1v
        del ds2v
        ds = xr.merge([ds_u.u10, ds_v.v10])
        print('merged')
        del ds_u
        del ds_v
        print("ds made")
        # 裁剪区域 todo
        deg = 1/8
        lon_st, lon_ed = 104, 124
        lat_st, lat_ed = 2, 22


        # 裁剪区域：经度 100~124，纬度 0~24
        cropped_ds = ds.sortby('latitude').sortby('longitude').sel(longitude=slice(lon_st, lon_ed), latitude=slice(lat_st, lat_ed))

        # 打印插值前的维度信息
        print(f"插值前维度 - 纬度: {len(cropped_ds.latitude)}, 经度: {len(cropped_ds.longitude)}")

        # 插值至 deg 度分辨率
        new_lon = np.arange(lon_st, lon_ed, deg)
        new_lat = np.arange(lat_st, lat_ed, deg)
        # new_lon = np.arange(100, 124, deg)
        # new_lat = np.arange(0, 24, deg)
        interpolated_ds = cropped_ds.interp(longitude=new_lon, latitude=new_lat, method='cubic')

        # 打印插值后的维度信息
        print(f"插值后维度 - 纬度: {len(interpolated_ds.latitude)}, 经度: {len(interpolated_ds.longitude)}")

        # 保存插值后的数据用于统计量计算
        all_u10_data.append(interpolated_ds['u10'])
        all_v10_data.append(interpolated_ds['v10'])

        if year == start_year:
            plot_interpolation_comparison(cropped_ds, interpolated_ds, output_dir)

        # 不再保存每年的单独文件，只保存到内存中用于统计计算
        print(f"✅ {year} 年数据处理完成")


    # 计算所有年份的统计量
    if all_u10_data and all_v10_data:
        print("正在计算统计量...")
        # 合并所有年份数据
        u10_combined = xr.concat(all_u10_data, dim='time')
        v10_combined = xr.concat(all_v10_data, dim='time')

        # 计算均值和标准差（跳过NaN值）
        # 直接在计算时使用时间切片
        time_slice = slice('1993-01-01', '2021-06-30')
        u10_mean = u10_combined.sel(time=time_slice).mean(skipna=True).compute()
        u10_std = u10_combined.sel(time=time_slice).std(skipna=True).compute()
        v10_mean = v10_combined.sel(time=time_slice).mean(skipna=True).compute()
        v10_std = v10_combined.sel(time=time_slice).std(skipna=True).compute()

        # 创建包含统计量的 Dataset
        stats_data = xr.Dataset({
            'u10_mean': u10_mean,
            'u10_std': u10_std,
            'v10_mean': v10_mean,
            'v10_std': v10_std
        })

        # 保存统计量
        stats_save_path = os.path.join(output_dir, 'wind_stats.nc')
        stats_data.to_netcdf(stats_save_path)
        print(f"✅ 统计量已保存至 {stats_save_path}")

        # 收集所有标准化后的数据用于合并
        all_normalized_datasets = []

        # 对每年的数据进行标准化处理并收集
        print("正在进行标准化处理...")
        
        # 标准化处理
        u10_normalized = (u10_combined - u10_mean) / u10_std
        v10_normalized = (v10_combined - v10_mean) / v10_std

        # 创建标准化后的 Dataset
        normalized_ds = xr.Dataset({
            'u10': u10_normalized,
            'v10': v10_normalized
        })

        # 保存合并后的数据集
        combined_output_path = os.path.join(output_dir, 'era5_wind_1_12_deg_all_years_normalized.nc')
        normalized_ds.to_netcdf(combined_output_path)
        print(f"✅ 所有年份合并数据已保存至 {combined_output_path}")

    print("所有年份数据处理完成！")


def plot_interpolation_comparison(original_ds, interpolated_ds, output_dir, date_index=0):
    """
    绘制插值前后对比图

    Parameters:
    original_ds (xr.Dataset): 插值前的数据
    interpolated_ds (xr.Dataset): 插值后的数据
    date_index (int): 选择绘图的时间索引
    """
    # 选择一天的数据进行对比
    u10_original = original_ds['u10'].isel(time=date_index)
    u10_interpolated = interpolated_ds['u10'].isel(time=date_index)

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 插值前
    im1 = axes[0].imshow(u10_original, extent=[u10_original.longitude.min(), u10_original.longitude.max(),
                                               u10_original.latitude.min(), u10_original.latitude.max()],
                         origin='lower', cmap='RdBu_r')
    axes[0].set_title(f'before\ntime: {str(u10_original.time.values)[:10]}')
    axes[0].set_xlabel('lon')
    axes[0].set_ylabel('lat')
    plt.colorbar(im1, ax=axes[0], label='10m U (m/s)')

    # 插值后
    im2 = axes[1].imshow(u10_interpolated, extent=[u10_interpolated.longitude.min(), u10_interpolated.longitude.max(),
                                                   u10_interpolated.latitude.min(), u10_interpolated.latitude.max()],
                         origin='lower', cmap='RdBu_r')
    axes[1].set_title(f'after\ntime: {str(u10_interpolated.time.values)[:10]}')
    axes[1].set_xlabel('lon')
    axes[1].set_ylabel('lat')
    plt.colorbar(im2, ax=axes[1], label='10m U (m/s)')

    plt.tight_layout()
    plt.savefig(output_dir+'/interpolation_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 插值前后对比图已保存为 interpolation_comparison.png")

# 主程序入口
if __name__ == "__main__":
    # 设置输入输出路径
    input_directory = ''
    processed_directory = ''

    # 批量处理多年数据
    process_era5_wind_data_with_normalization(input_directory, processed_directory, 1993, 2025)

