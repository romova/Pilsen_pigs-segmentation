# Segmentace cév prasete domácího ve snímcích z výpočetní tomografie
## Segmentation of Domestic Pig Vessels in Computed Tomography Images

Repozitář byl vytvořen v rámci diplomová práce, které se zaměřuje na návrh metody segmentace cév v dutině břišní u prasete domácího a její implementaci do běžně používaného medicínského vizualizačního nástroje 3DSlicer. Využitím segmentačních metod používaných v oblasti medicínského zobrazování poskytuje důležité informace o anatomických strukturách, čímž podporuje plánování chirurgických zákroků, předoperačních a pooperačních vyšetření a studium standardní anatomie prasete domácího.
Práce představuje anatomii cév v oblasti dutiny břišní. Dále se zaměřuje na různé architektury neuronových sítí, použité metriky a data sety, stejně jako na technické výzvy spojené se zobrazováním a zpracováním medicínských dat.
Výsledky experimentů ukazují, že navržené modely dosahují velmi dobré kvality při řešení úlohy segmentace cév v břišní dutině. Nejlepší navržený model dosahoval během validace hodnot 0,98 pro metriku Dice koeficient a 0,97 pro metriku IoU.


## Součástí repozitáře jsou  adresáře:
### Data
- obsahhují scripty pro získání veřejně dostupných datasetů, na kterých byly metody vyhodnoceny. A dataset Pilsen_Pigs, který byl vytvořen ve spolupráci c Lékařskou fakoultou Univerzity Karlovy.

### Kody a aplikace
- obsahuje implementace navržených metod modified 3D U-Net a 2D-to-3D U-Net a ukázkové kody s jejich využitím.

Ukázka segmetnace metodou 2D-to-3D U-Net na datasetu 3D-Ircad (zleva: původní snímek a maska cévního stromu, segmentace pozadí, segmentace popředí):

![2D-unet-cevy](https://github.com/user-attachments/assets/c573756f-607b-4963-ab73-5060b4b6874d)

Ukázka segmetnace metodou modified 3D U-Net na datasetu Deepvesselnet:

![3D_UNET_deep](https://github.com/user-attachments/assets/86bea882-2842-4333-8195-fd0db2693f51)

### 3D Slicer module
- obsahuje implementaci navržené metody do aplikace 3D Slicer a návod jak zpravoznit extension na vlastním PC.

![3D_UNET_ukazka](https://github.com/user-attachments/assets/180615a9-9349-4990-98d3-3db55f70fb9d)


