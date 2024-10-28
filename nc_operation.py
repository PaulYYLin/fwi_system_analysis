import xarray as xr
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import os
from dask.distributed import Client
from IPython.display import display, HTML


class nc_operation:
    def __init__(self, file_path_pattern, index_type, start_year, end_year, zarr_path="combined_data.zarr", extract_map=True):
       
        client = Client(n_workers=4)

        if os.path.exists(zarr_path):
            print("Loading data from existing Zarr file...")
            dataset = xr.open_zarr(zarr_path)
        else:
            print("Zarr file not found. Reading and combining NetCDF files...")
            dataset = xr.open_mfdataset(file_path_pattern, combine='by_coords', parallel=True, chunks={'Time': 365})

            # Datetime Transfer
            dataset = self.day_to_date(dataset, start_year, end_year)
            # Map Extraction
            dataset = self.extract_map(dataset)

            print("Saving combined data to Zarr format...")
            dataset.to_zarr(zarr_path, mode="w")

        # Assign dataset to class attribute
        self.data = dataset
        self.index_type = index_type


    def day_to_date(self, data, start_year, end_year):
        """
        Change "Time" values from 0-365 to a date range spanning from start_year-01-01 to end_year-12-31.
        
        Parameters:
            data (xarray.Dataset): The dataset with 'Time' in day-of-year format.
            start_year (int): The starting year of the data.
            end_year (int): The ending year of the data.
            
        Returns:
            xarray.Dataset: Dataset with 'Time' converted to datetime format.
        """
        # Create a list to hold datetime values
        date_times = []

        # Initialize day count
        day_count = 0

        # Loop over each year in the range and convert day-of-year to dates
        for year in range(start_year, end_year + 1):
            # Define the start date for the year
            start_date = f"{year}-01-01"
            
            # Determine the number of days in the year (leap year check)
            if pd.Timestamp(start_date).is_leap_year:
                days_in_year = 366
            else:
                days_in_year = 365
            
            # Extract the days for the current year
            year_data = data.isel(Time=slice(day_count, day_count + days_in_year))

            # Convert days of the year to dates for the current year
            year_dates = pd.to_datetime(pd.to_timedelta(year_data['Time'], unit='D') + pd.Timestamp(start_date))
            date_times.extend(year_dates)

            # Increment day count
            day_count += days_in_year

        # Assign the converted dates back to the 'Time' coordinate
        data['Time'] = date_times
        return data


    def extract_map(self, data, lat_range= [83, 41], lon_range= [219, 308]):
        '''
        Extract data with Latitude Range & Longitude Range
        Default is "Canada" with lat_range=(83, 41),  lon_range= (219, 308)
        '''
        extracted_data = data.sel(Latitude=slice(lat_range[0], lat_range[1]), Longitude=slice(lon_range[0], lon_range[1]))
        return extracted_data

    def show_map_for_date_range(self, resampled_data, start_year, start_month, end_year, end_month, output_folder="maps"):
        """
        Displays maps for each month within a specified date range based on resampled data.
        
        Parameters:
            resampled_data (xarray.DataArray): The resampled data containing ISI values and time information.
            start_year (int): The starting year for the range.
            start_month (int): The starting month for the range.
            end_year (int): The ending year for the range.
            end_month (int): The ending month for the range.
            output_folder (str): Folder to save individual map images.
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Convert the start and end dates to datetime objects
        start_date = pd.Timestamp(f"{start_year}-{start_month:02d}-01")
        end_date = pd.Timestamp(f"{end_year}-{end_month:02d}-01")

        # Create a list of monthly dates within the specified range
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')
        image_paths = []

        # Loop through each date in the specified range
        for selected_date in dates:
            # Select the ISI values for the specific month
            isi_values = resampled_data.sel(Time=selected_date, method="nearest")

            # Calculate map extent based on latitude and longitude
            minlat, maxlat = float(resampled_data['Latitude'].min()), float(resampled_data['Latitude'].max())
            minlongt, maxlongt = float(resampled_data['Longitude'].min()), float(resampled_data['Longitude'].max())

            # Plotting
            fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
            ax.set_extent([minlongt, maxlongt, minlat, maxlat], crs=ccrs.PlateCarree())

            # Add geographic features to the map
            ax.add_feature(cfeature.LAND, color='lightgray')
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=1, edgecolor='black')
            ax.add_feature(cfeature.STATES, linestyle='-', linewidth=0.5, edgecolor='gray')

            # Plot contour for ISI values
            contour = ax.contourf(
                isi_values['Longitude'], isi_values['Latitude'], isi_values,
                levels=20, cmap="viridis", transform=ccrs.PlateCarree()
            )

            # Add a colorbar to the map
            cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.01, aspect=25, label="ISI")
            cbar.ax.tick_params(labelsize=8)

            # Set the title to display the current date
            ax.set_title(f"Canada ISI - {selected_date.strftime('%Y-%m')}", fontsize=12)

            # Save each figure as an image file and store the path
            image_path = os.path.join(output_folder, f"map_{selected_date.strftime('%Y_%m')}.png")
            plt.savefig(image_path, bbox_inches='tight')
            image_paths.append(image_path)
            plt.close(fig)  # Close the figure to save memory

        # Display images in a row in the Jupyter Notebook
        html_content = "<div style='display: flex; justify-content: flex-start; flex-wrap: wrap;'>"
        for img_path in image_paths:
            html_content += f"<img src='{img_path}' style='margin-right: 10px; height: 200px;'/>"
        html_content += "</div>"

        # Display the HTML content
        display(HTML(html_content))

if __name__ == '__main__':

    INDEX_TYPE = 'ISI'
    nco = nc_operation(file_path_pattern=f'ISI/initial_spread_index_*.nc', index_type=INDEX_TYPE, start_year=2019, end_year=2020)
    canada_data = nco.extract_map(nco.data)
    monthly_isi = canada_data['ISI'].resample(Time='1ME').mean()
    nco.show_map_for_date_range(monthly_isi, 2019,5,2020,4)

