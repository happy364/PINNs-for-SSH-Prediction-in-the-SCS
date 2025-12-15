import os
import zipfile
import glob


def process_nc_files(folder_path):
    # 切换到目标文件夹
    os.chdir(folder_path)

    # 查找所有 .nc 文件
    nc_files = glob.glob("*.nc")

    for nc_file in nc_files:
        # 重命名文件
        zip_file = nc_file.replace('.nc', '.zip')
        os.rename(nc_file, zip_file)
        print(f"重命名: {nc_file} -> {zip_file}")

        # 解压文件
        try:
            extract_dir = zip_file.replace('.zip', '')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"解压完成: {zip_file} -> {extract_dir}/")
        except zipfile.BadZipFile:
            print(f"警告: {zip_file} 不是有效的 ZIP 文件")
        except Exception as e:
            print(f"解压 {zip_file} 时出错: {e}")


folder_path = r""
process_nc_files(folder_path)