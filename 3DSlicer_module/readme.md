# Návod: Vložení modulu `3D_Unet_vessels` do prostředí 3D Slicer

## 1. Stažení modulu
1. Stáhněte komprimovaný soubor `Pilsen_pigs_segmentator.zip` z GitHub repozitáře:  
2. Rozbalte celý obsah ZIP archivu.

## 2. Inicializace v 3D Sliceru
1. Otevřete aplikaci **3D Slicer**.
2. V horním menu vyberte:  
   `Edit` → `Application Settings` → záložka `Modules`.
3. V části `Additional module paths` klikněte na **Select directory** a vyberte složku `Pilsen_pigs_segmentator`.
4. Potvrďte volbu (dvakrát).
5. Aplikace vás vyzve k restartu — potvrďte a nechte ji znovu načíst.

## 3. Spuštění modulu
1. Po restartu se v záložce **Welcome to Slicer** zobrazí nová sekce:  
   `Pilsen_pigs_segmentator` → `3D_Unet_vessels`
2. Klikněte na název modulu — otevře se postranní panel s dvěma vstupními poli:
   - **Input volume**: nahrajte 3D snímek, který chcete segmentovat.
   - **Output volume**: zvolte existující výstupní objem, nebo klikněte na `Create new volume` pro vytvoření nového.

## 4. Spuštění segmentace
1. Klikněte na tlačítko `Apply`.
2. Segmentace se spustí — zpracování může trvat několik minut v závislosti na velikosti dat.
3. Po dokončení se výsledek objeví ve stromu objektů jako objemová data, která lze dále zpracovávat pomocí nástrojů 3D Sliceru.

---

🧠 **Tip:** Inicializační skript `run_me.txt` spusťte v interní Python konzoli Sliceru před prvním použitím modulu.
