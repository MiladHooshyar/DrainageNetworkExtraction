import Valley_Channel_Extraction

# TauDEM path:
TauDEM_path = "<TauDEM_folder>"

# Output folder
output_folder_path = "<output_folder>"
# Input DEM
DEM_file_path = "<Input_DEM_path>"
cell_size = 1.  # in DEM elevation unit
unit = 'm'  # ft or m

# Filtering parameter
num_iter = 50

# Valley connecting parameter
connect_ratio = 20

# Number of contours
number_contour = 30

# CH_ON : performing channel head identification
# CH_OFF : without channel head identification
option_channel_head = 'CH_OFF'

if __name__ == "__main__":
    Valley_Channel_Extraction.main(TauDEM_path, output_folder_path,
                                   DEM_file_path, cell_size, unit,
                                   num_iter, connect_ratio, number_contour,
                                   option_channel_head)
