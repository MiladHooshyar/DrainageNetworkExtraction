import Valley_Channel_Extraction

# TauDEM path:
TauDEM_path = "C:\\Program Files\\TauDEM\\TauDEM5Arc\\TauDEM Tools.tbx"

# Output folder
output_folder_path = "C:\\Data\\Codes\\Comb_Code\\Output"
# Input DEM
DEM_file_path = "C:\\Data\\Codes\\Comb_Code\\Input\\gro_DEM.tif"
cell_size = 0.5 # in DEM uint
unit = 'm' # ft or m

# Filtering paramters
num_iter = 50

# Valley connecting Paramter
connect_ratio = 20

# Number of contours
number_contour = 30

#CH_ON : perfrming channel head identification
#CH_OFF : without channel head identification
option_channel_head = 'CH_OFF' 


Valley_Channel_Extraction.main(TauDEM_path , output_folder_path , DEM_file_path, cell_size, unit, \
            num_iter, connect_ratio, number_contour , option_channel_head)
