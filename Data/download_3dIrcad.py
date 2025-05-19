import requests
import zipfile
import os

def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])

def safe_extract(zip_file, path="."):
    for member in zip_file.namelist():
        member_path = os.path.join(path, member)
        if not is_within_directory(path, member_path):
            raise Exception("Nebezpečný soubor v ZIP archivu!")
    zip_file.extractall(path)

def extract_zip(zip_path, extract_to):
    """Rozbalí ZIP soubor do zvolené složky a rekurzivně i všechny ZIPy uvnitř."""
    if not os.path.exists(zip_path):
        print(f"Soubor {zip_path} neexistuje!")
        return
    
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Rozbaluji {zip_path} do {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        safe_extract(zip_ref, extract_to)
    print(f"Rozbaleno: {zip_path}")

    # Rekurzivní rozbalení ZIPů uvnitř
    for root, _, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".zip"):
                nested_zip_path = os.path.join(root, file)
                nested_extract_to = os.path.splitext(nested_zip_path)[0]
                if not os.path.exists(nested_extract_to):
                    extract_zip(nested_zip_path, nested_extract_to)

# URL datasetu
url = "https://cloud.ircad.fr/index.php/s/JN3z7EynBiwYyjy/download"

# Cesty k souborům
dpath = "data"
os.makedirs(dpath, exist_ok=True)
output_file = os.path.join(dpath, "ircad_dataset.zip")

# Stahování datasetu
print("Stahuji dataset...")
response = requests.get(url, stream=True)

# Kontrola, zda bylo stahování úspěšné
if response.status_code == 200:
    with open(output_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Dataset byl úspěšně stažen jako {output_file}")

    # Rozbalení ZIPu + všech vnořených ZIPů
    extract_zip(output_file, dpath)

else:
    print(f"Chyba při stahování! Status code: {response.status_code}")
