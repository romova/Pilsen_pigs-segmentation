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
url = "https://cloud.ircad.fr/s/o7ggwdagy5Tfpf4/download"

# Cesty k souborům
dpath = "2D_to_3D/data/pigs"
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
    
    # Rozbalení souboru
    print("Rozbaluji dataset...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(dpath)
    print(f"Dataset byl úspěšně rozbalen do {dpath}")
else:
    print(f"Chyba při stahování! Status code: {response.status_code}")


# Příklad použití
zip_file = "2D_to_3D/data/pigs/ircad_dataset.zip"
destination_folder = "2D_to_3D/data/pilsen_pigs"
extract_zip(zip_file, destination_folder)