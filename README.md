## DrainageNetworkExtraction
This is a Python code to extract channel and valley networks for high resolution Digital Elevation Models (DEMs).

## A. Installation

The code itself does not require installation; however, there are some packages that are required to be installed before running the code.

1. “arcpy” library: this library is imbedded in ArcGIS package. If you install ArcGIS in your computer, the “arcpy” library and Python 2.7 will be installed automatically. It is recommended to use 64-bit python which is available on new versions of ArcGIS. 64-bit background geoprocessing is also available for older versions at http://resources.arcgis.com/en/help/main/10.1/index.html#/Background_Geoprocessing_64_bit/002100000040000000/.

2. “numpy”, “scipy” and “matplotlib” libraries: 64-bit version of these libraries can be found at (http://www.lfd.uci.edu/~gohlke/pythonlibs/). Make sure you download the libraries for Python 2.7. The whl files can be installed using pip as illustrated in this video (https://www.youtube.com/watch?v=zPMr0lEMqpo).

3. “TauDEM” toolbox: “TauDEM” can be downloaded from (http://hydrology.usu.edu/taudem/taudem5/downloads.html). By default, the installation path is (C:\Program Files\TauDEM). Before running the code, this path should be given as an input.


## B. Code structure

1. “Run.py” is the file for setting the parameters and running the code.

2. “Valley_Channel_Extraction.py” is the main code which calls the functions from “Channel_Fun.py” and “Valley_Fun.py” to delineate valley and channel networks.

3. “Valley_Fun.py” contains the functions for valley network delineation.

4. “Channel_Fun.py” includes the functions for channel head identification.

## C. Inputs

Before running the code, there are some parameters which should be set in “Run.py” including.

1. “TauDEM” toolbox folder path which is specified during the installation of “TauDEM” (TauDEM_path).

2. The path to the Output folder (output_folder_path).

3. The path to the DEM (DEM_file_path). Using tiff file is recommended. The code uses the spatial reference and the extent of the DEM for all other maps which are generated.

4. The pixel size of the DEM (cell_size).

5. The unit of the DEM (unit).

6. The number of iteration of the Perona-Malik nonlinear filter (num_iter). This parameter specifies the amount of smoothing on the DEM. The default value is 50.

7. Connecting parameter (connect_ratio). This parameter indirectly specifies the minimum length of gaps in the network. If the parameter increases, isolated segments are more likely to get connected to the network. The default value is 20 m.

8. The number of contours (number_contour). This specifies the number of contours for clustering in each tributary. The default value is 30.

9. The option for performing channel head identification (option_channel_head).  CH_ON : perfrming channel head identification, CH_OFF : without channel head identification


After setting the parameters, one can execute “Run.py” to extract the valley and channel networks. The output files will be saved in a folder called “maps” in the specified Output folder path.

## C. Publications

1. Hooshyar, M., Wang, D., Kim, S., Medeiros, S. C. and Hagen, S. C. (2016), Valley and channel networks extraction based on local topographic curvature and K-means clustering of contours. Water Resour. Res.. Accepted Author Manuscript. doi:10.1002/2015WR018479.


## D. Updates
1. Equation (1) in Hooshyar et al. (2016) in WRR is introduced as contour curvature which is consistant with the definition provided by Pelletier (2013). However, the accurate name for equation (1) would be tangential curvature as stated in Clubb (2014) and Mitasova and Hofierka (1993). 