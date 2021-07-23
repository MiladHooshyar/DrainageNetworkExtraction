import os
import shutil

import arcpy

import Channel_Fun
import Valley_Fun


def main(TauDEM_path, output_folder_path, DEM_file_path, cell_size, unit,
         num_iter, connect_ratio, number_contour, option_channel_head):
    arcpy.ImportToolbox(TauDEM_path)

    map_file_path = '%s' % output_folder_path + '\\maps'
    text_file_path = '%s' % output_folder_path + '\\texts'
    por_file_path = '%s' % output_folder_path + '\\pors'

    os.makedirs(map_file_path)
    os.makedirs(text_file_path)
    os.makedirs(por_file_path)
    os.makedirs(map_file_path + '\\' + 'Temp')

    arcpy.CheckOutExtension("Spatial")
    arcpy.env.scratchWorkspace = map_file_path + '\\' + 'Temp'
    arcpy.env.workspace = map_file_path
    arcpy.env.extent = DEM_file_path
    arcpy.env.outputCoordinateSystem = DEM_file_path
    if unit == 'm':
        scale = 1
    else:
        scale = 0.3048
    Valley_Fun.raster_copy(DEM_file_path, map_file_path + '\\Org_Dem.tif', scale)

    print('1. Filtering')
    Valley_Fun.Perona_malik(map_file_path + '\\Org_Dem.tif', map_file_path + '\\Smooth_Dem.tif', num_iter, cell_size,
                            'PM2')

    print('2. Computing Curvature')
    Valley_Fun.curvature(map_file_path + '\\Smooth_Dem.tif', map_file_path + '\\Curvature.tif',
                         map_file_path + '\\Curvature_profile.tif', cell_size, 'POLY_CON')

    print('3. Making Valley & Ridge Skeletons')
    sig_thresh = Valley_Fun.neg_bond(map_file_path + '\\Curvature.tif', map_file_path + '\Valley_Bond.tif',
                                     map_file_path + '\\Ridge_Bond.tif', map_file_path + '\\Conv_Sig.tif',
                                     map_file_path + '\\Dive_Sig.tif')

    print('4. Computing Curvature-Based Flow Direction')
    Valley_Fun.fill_dem(map_file_path + '\\Smooth_Dem.tif', map_file_path + '\\Fill_Dem.tif', 'GIS')
    Valley_Fun.flow_dir(map_file_path + '\\Fill_Dem.tif', map_file_path + '\\Fdir_inf.tif',
                        map_file_path + '\\Slope_inf.tif', 'DINF')
    Valley_Fun.Dinf_fix(map_file_path + '\\Fdir_inf.tif', map_file_path + '\\Curvature.tif',
                        map_file_path + '\\Modified_Fdir_inf.tif')
    Valley_Fun.flow_acc(map_file_path + '\\Modified_Fdir_inf.tif', map_file_path + '\\Modified_Facc_inf.tif', 'DINF')
    Valley_Fun.Dinf_to_D8(map_file_path + '\\Modified_Facc_inf.tif', map_file_path + '\\Modified_Fdir_8.tif',
                          map_file_path + '\\Modified_Facc_8.tif')
    Valley_Fun.cut_dem(map_file_path, map_file_path, 'Modified_Fdir_8.tif', 'Valley_Bond.tif',
                       'Cut_Modified_Fdir_8.tif')
    Valley_Fun.flow_acc_weigth(map_file_path + '\\Cut_Modified_Fdir_8.tif', map_file_path + '\\Curvature.tif',
                               map_file_path + '\\Cut_Modified_Facc_8.tif', 'D8')
    acc_tresh = 1 / float(connect_ratio)
    Valley_Fun.flow_acc_bond(map_file_path, map_file_path, acc_tresh, 'Cut_Modified_Facc_8.tif', 'Temp_Valley_00.tif')
    Valley_Fun.stream_order(map_file_path, map_file_path, 'Temp_Valley_00.tif', 'Cut_Modified_Fdir_8.tif',
                            'Temp_Order_00.tif')
    Valley_Fun.delet_isolated_stream(map_file_path, map_file_path, 'Temp_Valley_00.tif', 'Temp_Order_00.tif',
                                     'Modified_Fdir_8.tif', 'Modified_Facc_8.tif', 2, 2, 'Temp_Valley_000.tif')
    num_add = Valley_Fun.connect_stream_smart(map_file_path, map_file_path, 'Temp_Valley_000.tif',
                                              'Modified_Fdir_8.tif', 'Curvature.tif', 'Temp_Valley_10.tif',
                                              connect_ratio)

    print('5. Thinning & Connecting Valley Skeleton')
    trial = 0
    num_deleted = 1
    num_deleted_0 = 0
    num_add = 1
    num_add_0 = 0
    while num_add != num_add_0 and num_deleted != num_deleted_0:
        num_add_0 = num_add
        num_deleted_0 = num_deleted
        order_raster = 'Temp_Order_1' + '%s' % str(int(trial)) + '.tif'
        in_stream = 'Temp_Valley_1' + '%s' % str(int(trial)) + '.tif'
        out_stream = 'Temp_Valley_Thinned_1' + '%s' % str(int(trial)) + '.tif'
        con_out_stream = 'Temp_Valley_1' + '%s' % str(int(trial + 1)) + '.tif'
        Valley_Fun.stream_order(map_file_path, map_file_path, in_stream, 'Modified_Fdir_8.tif', order_raster)
        num_deleted = Valley_Fun.stream_delet_small_fast(map_file_path, map_file_path, 'Ridge_Bond.tif', in_stream,
                                                         'Fill_Dem.tif' \
                                                         , 'Modified_Fdir_8.tif', 'Modified_Facc_8.tif', order_raster,
                                                         'Curvature.tif', out_stream)
        num_add = Valley_Fun.connect_stream_smart(map_file_path, map_file_path, out_stream, 'Modified_Fdir_8.tif',
                                                  'Curvature.tif', con_out_stream, connect_ratio)
        trial = trial + 1

    print('6. Deleting Isolated Segments')
    in_stream = 'Temp_Valley_1' + '%s' % str(int(trial)) + '.tif'
    order_raster = 'Temp__Order_1' + '%s' % str(int(trial)) + '.tif'
    Valley_Fun.stream_order(map_file_path, map_file_path, in_stream, 'Modified_Fdir_8.tif', order_raster)
    Valley_Fun.delet_isolated_stream(map_file_path, map_file_path, in_stream, order_raster, 'Modified_Fdir_8.tif',
                                     'Modified_Facc_8.tif', 200, 2, 'Valley_Network.tif')

    for the_file in os.listdir(map_file_path):
        file_path = os.path.join(map_file_path, the_file)
        try:
            if os.path.isfile(file_path) and the_file.startswith("Temp"):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    Valley_Fun.stream_to_line(map_file_path, map_file_path, 'Valley_Network.tif', 'Modified_Fdir_8.tif',
                              'Valley_Network.shp')
    Valley_Fun.stream_order(map_file_path, map_file_path, 'Valley_Network.tif', 'Modified_Fdir_8.tif',
                            'Valley_Network_Order.tif')
    Valley_Fun.find_por_point(map_file_path, text_file_path, 'Valley_Network.tif', 'Fill_Dem.tif',
                              'Modified_Fdir_8.tif', 'Modified_Facc_8.tif', 'Valley_Network_Order.tif', 'por_point')

    if option_channel_head == 'CH_ON':
        print('7. Channel Head Extraction')
        head_ele = Channel_Fun.channel_head_find(por_file_path, map_file_path, text_file_path, 'Modified_Facc_8.tif',
                                                 'Modified_Fdir_8.tif', 'Curvature.tif', 'Fill_Dem.tif', 80, \
                                                 number_contour, 'contour_cluster', 0)
        arcpy.env.extent = DEM_file_path
        Valley_Fun.head_delete(map_file_path, map_file_path, 'Fill_Dem.tif', 'Valley_Network.tif',
                               'Modified_Fdir_8.tif', 'Valley_Network_Order.tif', head_ele, 'Channel_Network.tif')
        Valley_Fun.stream_to_line(map_file_path, map_file_path, 'Channel_Network.tif', 'Modified_Fdir_8.tif',
                                  'Channel_Network.shp')

        shutil.rmtree(text_file_path)
        shutil.rmtree(por_file_path)
    elif option_channel_head == 'CH_OFF':
        print('7. Channel Head Extraction is not requested')
