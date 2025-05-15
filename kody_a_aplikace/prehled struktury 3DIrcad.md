#     DICOM_file - NEW_voxelsize = [1,1,1]        ORIG_data_shape      ORIG_data_voxelsize   RESIZED_data_shape
# ----------------------------------------------------------------------------------------------------------------------------------
s1 = ['artery','portalvein','venoussystem']    # (3, 129, 512, 512)   # [1.6, 0.57, 0.57]   # (207, 292, 292)
s2 = ['portalvein','venacava']                 # (2, 172, 512, 512)   # [1.6, 0.78, 0.78]   # (276, 401, 401)
#s3 = ['portalvein','venacava'] # data = maska # (2, 200, 512, 512)   # [1.2, 0.62, 0.62]   # (250, 320, 320)
s4 = ['artery','portalvein','venoussystem']    # (3, 91,  512, 512)   # [2.0, 0.74, 0.74]   # (182, 381, 381)
s5 = ['artery','portalvein','venoussystem']    # (3, 139, 512, 512)   # [1.6, 0.78, 0.78]   # (223, 401, 401)
s6 = ['artery','portalvein','venoussystem']    # (3, 135, 512, 512)   # [1.6, 0.78, 0.78]   # (217, 401, 401)
s7 = ['artery','portalvein','venoussystem']    # (3, 151, 512, 512)   # [1.6, 0.78, 0.78]   # (242, 401, 401)
s8 = ['artery','portalvein','venoussystem']    # (3, 124, 512, 512)   # [1.6, 0.56, 0.56]   # (199, 288, 288)
s9 = ['artery','portalvein','venoussystem']    # (3, 111, 512, 512)   # [2.0, 0.87, 0.87]   # (222, 448, 448)
# s10 = ['venacava', 'portalvein'] # Problem  'portalvein1',   # (1, 225, 512, 512)   # [1.6, 0.73, 0.73]   # (196, 377, 377)
s11 = ['artery','portalvein','venacava']       # (3, 132, 512, 512)   # [1.6, 0.72, 0.72]   # (212, 369, 369)
s12 = ['artery','portalvein','venacava']       # (3, 260, 512, 512)   # [1.0, 0.68, 0.68]   # (260, 349, 349)
s13 = ['artery','portalvein','venacava']       # (3, 122, 512, 512)   # [1.6, 0.67, 0.67]   # (196, 344, 344)
s14 = ['portalvein','venoussystem']                # (2, 113, 512, 512)   # [1.6, 0.72, 0.72]   # (181, 369, 369)
s15 = ['portalvein','venoussystem']            # (2, 125, 512, 512)   # [1.6, 0.78, 0.78]   # (201, 401, 401)
s16 = ['portalvein','venoussystem']            # (2, 155, 512, 512)   # [1.6, 0.69, 0.69]   # (249, 358, 358)
s17 = ['artery','portalvein','venoussystem']   # (3, 119, 512, 512)   # [1.6, 0.74, 0.74]   # (191, 381, 381)
s18 = ['portalvein','venacava']                # (2, 74,  512, 512)   # [2.5, 0.74, 0.74]   # (185, 381, 381)
s19 = ['portalvein','venoussystem']            # (2, 124, 512, 512)   # [4.0, 0.70, 0.70]   # (496, 360, 360)
# s20 = ['artery','portalvein','venacava']# spatne segmentovano       # (3, 225, 512, 512)   # [2.0, 0.81, 0.81]   # (450, 415, 415)

portalvein = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]     # portalvein - zdola do jater
artery = [1, 4, 5, 6, 7, 8, 9, 11, 12, 13, 17] 
venacava = [2, 11, 12, 13, 18]                                      # venacava - shora do jater
venoussystem = [1, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19]
cevy1 = [1, 4, 5, 6, 7, 8, 9, 17]      # artery, portalvein, venoussystem
cevy2 = [2, 11, 12, 13, 18]   # portalvein, venacava
cevy3 = [1, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 19]   # portalvein, venoussystem

s_artery = ['artery','artery'] # format [jmeno_modelu, slozky ...]
s_portalvein = ['portalvein', 'portalvein']
s_venacava = ['venacava', 'venacava']
s_venoussystem = ['venoussystem', 'venoussystem']
s_cevy1 = ['cevy1', 'artery','portalvein','venoussystem']  
s_cevy2 = ['cevy2', 'portalvein','venacava']    
s_cevy3 = ['cevy3', 'portalvein','venoussystem'] 