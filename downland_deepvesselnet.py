import os
import zipfile
import requests

def mywget(url, filename):
    """Downloads a file from the given URL and saves it as filename."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
    
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Downloaded {filename}")
    
def unzip_all(zip_dir):
    """Unzips all zip files in the specified directory without creating additional folders."""
    for file in os.listdir(zip_dir):
        if file.endswith('.zip'):
            zip_path = os.path.join(zip_dir, file)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_dir)
                print(f'Extracted {file} into {zip_dir}')
                
# Example usage
URL_raw = "https://syncandshare.lrz.de/dl/fiA6KKF6AAzux7Pw2ANnBnLf"
URL_seg = "https://syncandshare.lrz.de/dl/fiBtJdposDALJ3GwV2GfT15A"
URL_bif = "https://syncandshare.lrz.de/dl/fiGhETaSTNuh52VwYXhmfjAU"
URL_centerline = "https://syncandshare.lrz.de/dl/fiAWNW6tFtw5Ppoeg4Qr2Ufo"
URL_points = "https://syncandshare.lrz.de/dl/fi2w4j7CzsNGqoRv7VkSq7k3"
URL_radius = "https://syncandshare.lrz.de/dl/fiXUGQ5rjKtzMbtekPorVy1t"

mywget(URL_raw, "data/deepvesselnet/raw.zip")
mywget(URL_seg, "data/deepvesselnet/seg.zip")
mywget(URL_bif, "data/deepvesselnet/bif.zip")
mywget(URL_points, "data/deepvesselnet/points.zip")
mywget(URL_radius, "data/deepvesselnet/radius.zip")
mywget(URL_centerline, "data/deepvesselnet/centerline.zip")

unzip_all(os.getcwd()+ "/data/deepvesselnet")
print("Done")