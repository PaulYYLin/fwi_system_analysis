import os
import xarray as xr
import pandas as pd
import dask
from dask.distributed import Client
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from IPython.display import display, HTML

class nc_operation:
    def __init__(self, file_path_pattern, index_type, start_year, end_year, lat_range=[60, 49], lon_range=[245, 255], output_zarr_path="combined_data.zarr"):
        """
        初始化 nc_operation 類別，並進行文件合併與處理。
        
        :param file_path_pattern: 每年 NetCDF 文件的路徑模式
        :param index_type: 指標類型，例如 'ISI'
        :param start_year: 起始年份
        :param end_year: 結束年份
        :param lat_range: 緯度範圍，預設為 [60,49] Alberta
        :param lon_range: 經度範圍，預設為 [245, 255]  Alberta
        :param output_zarr_path: 輸出的 Zarr 文件路徑
        """
        client = Client(n_workers=4)
        self.index_type = index_type
        self.file_path_pattern = file_path_pattern
        self.start_year = start_year
        self.end_year = end_year
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.output_zarr_path = output_zarr_path

        if os.path.exists(output_zarr_path):
            print("Loading data from existing Zarr file...")
            self.data = xr.open_zarr(output_zarr_path)
        else:
            self.data = self.load_and_process_data()
            
            # Rechunking to ensure uniform chunk sizes across the dataset
            self.data = self.data.chunk({'Time': 365})
            self.data.to_zarr(output_zarr_path, mode="w", consolidated=True)

    def load_and_process_data(self):
        """
        逐年讀取並處理每個 NetCDF 文件，將其轉換為日期格式，擷取指定範圍，並合併所有年份數據。
        
        :return: 合併後的 xarray Dataset
        """
        yearly_datasets = []

        for year in range(self.start_year, self.end_year + 1):
            file_path = self.file_path_pattern.replace("*", str(year))
            if os.path.exists(file_path):
                print(f"Processing file: {file_path}")

                # 讀取並設定時間分塊以便處理
                dataset = xr.open_dataset(file_path, chunks={'Time': 365})
                
                # 日期轉換
                dataset = self.day_to_date(dataset, year)
                
                # 經緯度範圍擷取
                dataset = dataset.sel(Latitude=slice(self.lat_range[0], self.lat_range[1]),
                                      Longitude=slice(self.lon_range[0], self.lon_range[1]))

                # 加入到年度數據列表
                yearly_datasets.append(dataset)

        # 合併所有年度數據
        combined_data = xr.concat(yearly_datasets, dim="Time")
        return combined_data

    def day_to_date(self, data, year):
        """
        將 Time 從 0-365 轉換成特定年份的日期格式。
        
        :param data: xarray Dataset
        :param year: 年份
        :return: 日期轉換後的 xarray Dataset
        """
        start_date = pd.Timestamp(f"{year}-01-01")
        dates = pd.date_range(start=start_date, periods=len(data['Time']), freq='D')
        data = data.assign_coords(Time=dates)
        return data

    def show_map_for_date_range(self, resampled_data, start_year, start_month, end_year, end_month, output_folder="maps"):
        """
        顯示指定日期範圍的每月地圖。
        
        :param resampled_data: 包含 ISI 值和時間信息的 xarray DataArray
        :param start_year: 起始年
        :param start_month: 起始月
        :param end_year: 結束年
        :param end_month: 結束月
        :param output_folder: 用於儲存個別地圖圖像的資料夾
        """
        # 創建輸出目錄（若不存在）
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 設置起始和結束日期
        start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01")
        end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01")
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        image_paths = []

        # 繪製每個日期的地圖
        for selected_date in dates:
            isi_values = resampled_data.sel(Time=selected_date, method="nearest")
            minlat, maxlat = float(resampled_data['Latitude'].min()), float(resampled_data['Latitude'].max())
            minlongt, maxlongt = float(resampled_data['Longitude'].min()), float(resampled_data['Longitude'].max())

            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([minlongt, maxlongt, minlat, maxlat], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, color='lightgray')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, edgecolor='black')
            ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor='gray')

            contour = ax.contourf(
                isi_values['Longitude'], isi_values['Latitude'], isi_values,
                levels=20, cmap="viridis", transform=ccrs.PlateCarree()
            )
            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.01, aspect=25, label="ISI")
            cbar.ax.tick_params(labelsize=8)
            ax.set_title(f"Canada ISI - {selected_date.strftime('%Y-%m')}", fontsize=12)

            image_path = os.path.join(output_folder, f"map_{selected_date.strftime('%Y_%m')}.png")
            plt.savefig(image_path, bbox_inches='tight')
            image_paths.append(image_path)
            plt.close(fig)
        html_content = "<div style='display: flex; justify-content: flex-start; flex-wrap: wrap; '>"
        for img_path in image_paths:
            html_content += f"<img src='{img_path}' style='margin-right: 10px; height: 200px;'/>"
        html_content += "'</divy"
        display (HTML (html_content) )
