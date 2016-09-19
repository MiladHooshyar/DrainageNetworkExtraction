import Valley_Channel_Extraction

# TauDEM path:
TauDEM_path = "C:\\Program Files\\TauDEM\\TauDEM5Arc\\TauDEM Tools.tbx"

# Output folder
output_folder_path = "Z:\\ACTIVE\\MiladHooshyar\\Input_Angel\\Indian_Creek\\Output_Valley"
# Input DEM
DEM_file_path = "C:\\Data\\Output\\Channel_Paper\\IC\\maps\\0\\OD.tif"
cell_size = 1 # in DEM uint
unit = 'm' # ft or m

# Filtering paramters
num_iter = 50

# Valley connecting Paramter
connect_ratio = 20

# Number of contours
number_contour = 30


Valley_Channel_Extraction.main(TauDEM_path , output_folder_path , DEM_file_path, cell_size, unit, \
            num_iter, connect_ratio, number_contour)
