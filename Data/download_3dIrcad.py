import requests
import zipfile
import os

def extract_zip(zip_path, extract_to):
    """Rozbalí ZIP soubor do zvolené složky."""
    if not os.path.exists(zip_path):
        print(f"Soubor {zip_path} neexistuje!")
        return
    
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Rozbaluji {zip_path} do {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Rozbalení dokončeno.")

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
    
    # Bezpečné rozbalení ZIP souboru, včetně vnořených adresářů
    print("Rozbaluji dataset...")

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

    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        safe_extract(zip_ref, dpath)

    print(f"Dataset byl úspěšně rozbalen do {dpath}")
else:
    print(f"Chyba při stahování! Status code: {response.status_code}")


# Příklad použití
zip_file = "ircad_dataset.zip"
destination_folder = "ircad_dataset"
extract_zip(zip_file, destination_folder)
