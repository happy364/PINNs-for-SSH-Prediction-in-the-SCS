import cdsapi
import os

output_dir = ''
os.makedirs(output_dir, exist_ok=True)

c = cdsapi.Client()

# 分年下载，每年一个请求
years = [str(year) for year in range(1993, 2026)]

for year in years:
    filename = f'era5_daily_mean_u10_v10_{year}'
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        print(f"文件 {filename} 已存在，跳过...")
        continue

    print(f"正在下载 {year} 年的日平均数据...")

    dataset = "derived-era5-single-levels-daily-statistics"
    request1 = {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": year,
            "month": [
                "01", "02", "03", "04", "05", "06",

            ],
            "day": [
                "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
            ],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "6_hourly",
        }
    request2 = {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind"
            ],
            "year": year,
            "month": [

                "07", "08", "09", "10", "11", "12"
            ],
            "day": [
                "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
                "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
            ],
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "6_hourly",
        }


    c.retrieve(
        dataset,
        request1
        ,
        output_path+'_1.nc')
    c.retrieve(
        dataset,
        request2
        ,
        output_path+'_2.nc')
    print(f"✓ {year} 年日平均数据下载完成")

print("所有年份日平均数据下载任务已完成！")
