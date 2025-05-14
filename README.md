# Segmentation of Domestic Pig Vessels in Computed Tomography Images
## Segmentation of Domestic Pig Vessels in Computed Tomography Images

Repozitář byl vytvořen v rámci diplomová práce, které se zaměřuje na návrh metody segmentace cév v dutině břišní u prasete domácího a její implementaci do běžně používaného medicínského vizualizačního nástroje 3DSlicer. Využitím segmentačních metod používaných v oblasti medicínského zobrazování poskytuje důležité informace o anatomických strukturách, čímž podporuje plánování chirurgických zákroků, předoperačních a pooperačních vyšetření a studium standardní anatomie prasete domácího.
Práce představuje anatomii cév v oblasti dutiny břišní. Dále se zaměřuje na různé architektury neuronových sítí, použité metriky a data sety, stejně jako na technické výzvy spojené se zobrazováním a zpracováním medicínských dat.
Výsledky experimentů ukazují, že navržené modely dosahují velmi dobré kvality při řešení úlohy segmentace cév v břišní dutině. Nejlepší navržený model dosahoval během validace hodnot 0,98 pro metriku Dice koeficient a 0,97 pro metriku IoU.
![3D_UNET_ukazka](https://github.com/user-attachments/assets/180615a9-9349-4990-98d3-3db55f70fb9d)

Součástí repozitáře jsou dva adresáře:

### Kody a aplikace
Adresář obsahuje implementace navržených metod a ukázkové kody s jejich využitím.

kody_a_aplikace:.
+---3DSlicer_module
|       Pilsen_pigs_segmentator.zip
|
+---Analýza
|   |   original_environment.txt
|   |   prehled struktury 3DIrcad.md
|   |   technicke_problemy.ipynb
|   |   Total_segmentator_example_Tx017D_Art.nii.gz.zip
|   |
|   +---2D_to_3D
|   |   |   2d_to3D_train.ipynb
|   |   |   2D_to_3D_predict.ipynb
|   |   |   methods_2D.py
|   |   |
|   |   \---model_test
|   |           3DIrcad_cevy1.zip
|   |           3dIrcad_liver.zip
|   |           3DIrcad_ven.zip
|   |           deepvesselnet.zip
|   |           pilsen_pigs.zip
|   |
|   +---3D_modif
|   |       methods.py
|   |       UNET_3D_classic.ipynb
|   |       UNET_3D_modified.ipynb
|   |       UNET_3D_segment_and_transform.ipynb
|   |       models.zip
|   |       models_transformed.zip
|   |
|   \---libraries
|           classic_unets.py
|           ndnoise.zip
|
\---Data
        downland_3dIrcad.py
        downland_deepvesselnet.py

## 3D Slicer module
Adresář obsahuje implementaci navržené metody do aplikace 3D Slicer a návod jak zpravoznit extension na vlastním PC.

|Pilsen_pigs_segmentator.zip:.
|   CMakeLists.txt
|   methods.py
|   run_me.txt
|   segment.py
|
+---.vs
|   |   CMake Overview
|   |   ProjectSettings.json
|   |   slnx.sqlite
|   |   VSWorkspaceState.json
|   |
|   \---segment
|       +---FileContentIndex
|       |       96f3faf1-8498-4ebf-acdf-cce554d3c9e1.vsidx
|       |
|       \---v17
|               DocumentLayout.json
|
+---modely
|       pisen_pigs_small_cycle_76.h5
|
+---out
|   \---build
|       \---x64-Debug
|           |   VSInheritEnvironments.txt
|           |
|           +---.cmake
|           |   \---api
|           |       \---v1
|           |           \---query
|           |               \---client-MicrosoftVS
|           |                       query.json
|           |
|           \---CMakeFiles
|               |   CMakeConfigureLog.yaml
|               |
|               +---3.29.5-msvc4
|               |   |   CMakeCCompiler.cmake
|               |   |   CMakeCXXCompiler.cmake
|               |   |   CMakeDetermineCompilerABI_C.bin
|               |   |   CMakeRCCompiler.cmake
|               |   |   CMakeSystem.cmake
|               |   |
|               |   +---CompilerIdC
|               |   |   |   CMakeCCompilerId.c
|               |   |   |   CMakeCCompilerId.exe
|               |   |   |   CMakeCCompilerId.obj
|               |   |   |
|               |   |   \---tmp
|               |   \---CompilerIdCXX
|               |       |   CMakeCXXCompilerId.cpp
|               |       |   CMakeCXXCompilerId.exe
|               |       |   CMakeCXXCompilerId.obj
|               |       |
|               |       \---tmp
|               +---CMakeScratch
|               |   \---TryCompile-f07f79
|               |       |   .ninja_deps
|               |       |   .ninja_lock
|               |       |   .ninja_log
|               |       |   build.ninja
|               |       |   CMakeCache.txt
|               |       |   CMakeLists.txt
|               |       |   cmake_install.cmake
|               |       |   cmTC_b332b.exe
|               |       |   cmTC_b332b.ilk
|               |       |   cmTC_b332b.pdb
|               |       |   cmTC_b332b_loc
|               |       |
|               |       \---CMakeFiles
|               |           |   cmake.check_cache
|               |           |   rules.ninja
|               |           |   TargetDirectories.txt
|               |           |
|               |           +---cmTC_b332b.dir
|               |           |       CMakeCXXCompilerABI.cpp.obj
|               |           |       embed.manifest
|               |           |       intermediate.manifest
|               |           |       manifest.rc
|               |           |       manifest.res
|               |           |       vc140.pdb
|               |           |
|               |           \---pkgRedirects
|               +---pkgRedirects
|               \---ShowIncludes
|                       foo.h
|                       main.c
|                       main.obj
|
+---Resources
|   +---Icons
|   |       segment.png
|   |
|   \---UI
|           segment.ui
|
+---Segmentator
+---Testing
|   |   CMakeLists.txt
|   |
|   \---Python
|           CMakeLists.txt
|
\---__pycache__
        methods.cpython-39.pyc
        segment.cpython-39.pyc
