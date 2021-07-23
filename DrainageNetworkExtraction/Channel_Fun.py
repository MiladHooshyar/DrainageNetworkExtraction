import math
import os
import sys

import arcpy
import matplotlib.pyplot as pl
import numpy as np
from arcpy import env, Raster
from arcpy.sa import *
from numpy import linspace, random
from scipy.interpolate import splprep, splev

arcpy.CheckOutExtension("Spatial")


##########################################################################
def channel_head_find(por_file_path, map_file_path, text_file_path, flow_acc_raster, flow_dir_raster, curve_Raster,
                      dem_raster, number_of_uni_point, number_contour, method, flag_plot):
    por_point, number_of_por = read_por_point(text_file_path)

    env.workspace = por_file_path
    flow_acc = '%s' % map_file_path + '/' + '%s' % flow_acc_raster
    flow_dir = '%s' % map_file_path + '/' + '%s' % flow_dir_raster
    curve_Raster = '%s' % map_file_path + '/' + '%s' % curve_Raster
    raster = '%s' % map_file_path + '/' + '%s' % dem_raster

    curve_Array = arcpy.RasterToNumPyArray(curve_Raster)

    dsc = arcpy.Describe(curve_Raster)
    X_min = dsc.EXTENT.XMin
    Y_min = dsc.EXTENT.YMin

    X_max = dsc.EXTENT.XMax
    Y_max = dsc.EXTENT.YMax

    dy = dsc.meanCellHeight
    dx = dsc.meanCellWidth

    out_path = por_file_path
    geometry_type = "POINT"
    has_m = "DISABLED"
    has_z = "DISABLED"

    print('Total number of heads', number_of_por)

    Ele_Thresh = np.zeros((number_of_por, 1))

    for p in range(0, number_of_por):

        sys.stdout.write("\r%d-" % p)
        num_curve = 0
        x_data = np.zeros((1, number_of_uni_point))
        y_data = np.zeros((1, number_of_uni_point))
        c_data = np.zeros((1, number_of_uni_point))
        por_data = np.zeros((1, 1))
        org_x_data = np.zeros((1, number_of_uni_point))
        org_y_data = np.zeros((1, number_of_uni_point))
        sorg_x_data = np.zeros((1, number_of_uni_point))
        sorg_y_data = np.zeros((1, number_of_uni_point))

        if flag_plot == 1:
            fig = pl.figure(0)
            ax = fig.add_subplot(111)
        # making contour lines
        arcpy.env.extent = raster

        out_name = 'por_' + str(p) + '.shp'
        arcpy.CreateFeatureclass_management(out_path, out_name, geometry_type, "", has_m, has_z, dsc.SpatialReference)
        cursor = arcpy.da.InsertCursor(out_name, ["SHAPE@"])
        cursor.insertRow([arcpy.Point(float(por_point[p][0]), float(por_point[p][1]))])
        del cursor
        tolerance = 0
        outSnapPour = SnapPourPoint(out_name, flow_acc, tolerance)

        small_basin = Watershed(flow_dir, outSnapPour)

        basin_Array = arcpy.RasterToNumPyArray(small_basin)

        poly_bond = '%s' % por_file_path + '/' + 'bond' + str(p) + '.shp'
        buff_poly_bond = '%s' % por_file_path + '/' + 'bond_buffer' + str(p) + '.shp'

        arcpy.RasterToPolygon_conversion(small_basin, poly_bond)
        arcpy.Buffer_analysis(poly_bond, buff_poly_bond, "3 Meters")

        subbasin_B = '%s' % por_file_path + '/subDEM_B_' + str(p) + '.tif'
        arcpy.Clip_management(raster, "", subbasin_B, buff_poly_bond, "", "ClippingGeometry")
        subbasin = Con(Raster(subbasin_B) <= por_point[p][2], Raster(subbasin_B))
        subbasin.save('%s' % por_file_path + '/subDEM_' + str(p) + '.tif')

        contour_interval = round((por_point[p][2] - por_point[p][3]) / number_contour, 2)

        arcpy.env.extent = buff_poly_bond

        if contour_interval >= 0.01:
            make_contour(subbasin, 'basin_' + str(p), por_file_path, contour_interval)
            number_of_ID, ID_elevation = read_point_data('basin_' + str(p) + '_point', por_file_path, por_file_path)
        else:
            number_of_ID = 0

        outSnapPour = None
        subbasin = None
        small_basin = None

        if number_of_ID <= 5:
            print(p, '      Small initial of final contour')
        else:
            x_0 = 0
            y_0 = 0
            temp_x = np.zeros((number_of_ID, number_of_uni_point))
            temp_y = np.zeros((number_of_ID, number_of_uni_point))
            sorted_x = np.zeros((number_of_ID, number_of_uni_point))
            sorted_y = np.zeros((number_of_ID, number_of_uni_point))

            for ID in range(0, number_of_ID):
                temp_kapa_1, x_0, y_0, x_p, y_p, x, y = poly_fit(ID, 'basin_' + str(p) + '_point', por_file_path,
                                                                 por_file_path, number_of_uni_point, 0, x_0, y_0)
                for i in range(0, number_of_uni_point):
                    temp_x[ID, i] = x[i]
                    temp_y[ID, i] = y[i]

            sorted_ID_elevation = sorted(ID_elevation, key=lambda temp_x: temp_x[1], reverse=True)
            for ID in range(0, number_of_ID):
                org_ID = sorted_ID_elevation[ID][0]
                sorted_x[ID, :] = temp_x[org_ID, :]
                sorted_y[ID, :] = temp_y[org_ID, :]

            org_x, org_y, ID_elevation, number_of_ID = contour_delete(sorted_x, sorted_y, sorted_ID_elevation,
                                                                      number_of_ID, number_of_uni_point,
                                                                      contour_interval, por_point[p][2],
                                                                      por_point[p][3])
            org_x, org_y = contour_direction_fix(org_x, org_y, ID_elevation, number_of_ID, number_of_uni_point)

            for ID in range(0, number_of_ID):
                org_x_data = np.copy(np.append(org_x_data, org_x[ID, :].reshape(1, number_of_uni_point), axis=0))
                org_y_data = np.copy(np.append(org_y_data, org_y[ID, :].reshape(1, number_of_uni_point), axis=0))

            c = contour_curve(org_x, org_y, curve_Array, number_of_uni_point, number_of_ID, X_min, Y_max, dx, dy)

            for ID in range(0, number_of_ID):
                x_data = np.copy(np.append(x_data, org_x[ID, :].reshape(1, number_of_uni_point), axis=0))
                y_data = np.copy(np.append(y_data, org_y[ID, :].reshape(1, number_of_uni_point), axis=0))
                c_data = np.copy(np.append(c_data, c[ID, :].reshape(1, number_of_uni_point), axis=0))
                por_data = np.copy(np.append(por_data, p * np.ones((1, 1)), axis=0))
                num_curve = num_curve + 1

        for the_file in os.listdir(por_file_path):
            file_path = os.path.join(por_file_path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception, e:
                print(e)

        x_data = np.delete(x_data, 0, 0)
        y_data = np.delete(y_data, 0, 0)
        c_data = np.delete(c_data, 0, 0)
        por_data = np.delete(por_data, 0, 0)

        org_x_data = np.delete(org_x_data, 0, 0)
        org_y_data = np.delete(org_y_data, 0, 0)

        sorg_x_data = np.delete(sorg_x_data, 0, 0)
        sorg_y_data = np.delete(sorg_y_data, 0, 0)

        number_of_k = 2
        number_of_ID = x_data.shape[0]
        min_contour = max(int(number_of_ID * 0.1), 2)

        if num_curve <= 5:
            print('     Small number of final contour')
        else:

            if flag_plot == 1:
                for ID in range(0, num_curve - 1):
                    pl.plot(org_x_data[ID, :], org_y_data[ID, :], 'g', linewidth=2)
                    ax.annotate(round(ID_elevation[ID, 1]), xy=(org_x_data[ID, 0] - 5, org_y_data[ID, 0] - 5))

            ele_error = np.zeros((number_of_ID, 2))
            for initial_tresh in range(0, number_of_ID):
                initial_cluster_code = np.zeros((1, number_of_ID))

                for ID in range(0, number_of_ID):
                    if ID > initial_tresh:
                        initial_cluster_code[0][ID] = 1
                if method == 'contour_cluster':
                    x_ave, y_ave = cluster_average(initial_cluster_code, x_data, y_data, number_of_k, number_of_ID,
                                                   number_of_uni_point)
                    err = cluster_performance(x_data, y_data, initial_cluster_code, x_ave, y_ave, number_of_k,
                                              number_of_ID, number_of_uni_point)
                    ele_error[initial_tresh, :] = [ID_elevation[initial_tresh, 1], err]

            if flag_plot == 1:
                fig = pl.figure(1)
                pl.plot(ele_error[:, 0], ele_error[:, 1], 'ro-')
                for i in range(0, ele_error.shape[0]):
                    print
                    ele_error[i, 0], ele_error[i, 1]

            length_error = find_local_min(ele_error)

            ele_tresh = 0
            count = 0
            for i in range(0, number_of_ID):
                if length_error[i, 0] > number_of_ID * 0.1:
                    ele_tresh = ele_tresh + ID_elevation[i, 1] * length_error[i, 0]
                    count = count + length_error[i, 0]

            if count == 0:
                print(p, 'not dominant cluster found')
            else:
                ele_tresh = ele_tresh / count
                Ele_Thresh[por_data[ID, 0], 0] = ele_tresh

    return Ele_Thresh


def find_local_min(ele_error):
    num_p = ele_error.shape[0]
    length_error = np.zeros((num_p, 2))
    count = 1
    for i in range(0, num_p - 1):
        if ele_error[i + 1, 1] >= ele_error[i, 1]:
            if count > 1:
                length_error[i, :] = [count, ele_error[i, 1]]
            count = 1
        else:
            count = count + 1
    count = 0
    for i in range(num_p - 1, 0, -1):
        if ele_error[i - 1, 1] >= ele_error[i, 1]:
            length_error[i, 0] = length_error[i, 0] + count
            count = 0
        else:
            count = count + 1

    return length_error


def add_mapped_headpoints(in_file_path, number_of_por_point):
    f = open('%s' % in_file_path + '/extracted_heads.txt')
    head_point = []
    for line in f:
        head_point.append([float(temp_xx) for temp_xx in line.split()])
    f.close()

    number_of_point = len(head_point)

    for p in range(0, number_of_point):
        pl.plot(head_point[p][0], head_point[p][1], 'bo')

    head_of_por = np.zeros((number_of_por_point, 1))
    for por in range(0, number_of_por_point):
        head_of_por[por][0] = -1

    for p in range(0, number_of_point):
        por = int(head_point[p][2])
        if por >= 0:
            head_of_por[por][0] = p

    return head_of_por, head_point


def read_por_point(in_file_path):
    f = open('%s' % in_file_path + '/por_point.txt')
    por_point = []
    for line in f:
        por_point.append([float(temp_xx) for temp_xx in line.split()])
    f.close()

    number_of_por = len(por_point)

    return por_point, number_of_por


########################################################################################
########################################################################################

def make_contour(file_raster, name, out_file_path, contour_interval):
    file_contour = '%s' % out_file_path + '/' + '%s' % name + '_con.shp'

    min_elevation = arcpy.GetRasterProperties_management(file_raster, "MINIMUM")
    Contour(file_raster, file_contour, contour_interval, 0, 1)

    file_contour_point = '%s' % out_file_path + '/' + '%s' % name + '_point.shp'
    arcpy.FeatureVerticesToPoints_management(file_contour, file_contour_point, "ALL")

    inFeatures = file_contour_point
    fieldName1 = "x_degree"
    fieldName2 = "y_degree"
    fieldPrecision = 18
    fieldScale = 11
    arcpy.AddField_management(inFeatures, fieldName1, "DOUBLE", fieldPrecision, fieldScale)
    arcpy.AddField_management(inFeatures, fieldName2, "DOUBLE", fieldPrecision, fieldScale)

    arcpy.CalculateField_management(inFeatures, fieldName1, "!shape.firstpoint.X!", "PYTHON_9.3")
    arcpy.CalculateField_management(inFeatures, fieldName2, "!shape.firstpoint.Y!", "PYTHON_9.3")

    all_cp = arcpy.da.TableToNumPyArray(inFeatures, ('ID', 'CONTOUR', 'x_degree', 'y_degree'))

    os.remove('%s' % out_file_path + '/' + '%s' % name + '_con.shp')
    os.remove('%s' % out_file_path + '/' + '%s' % name + '_point.shp')
    os.remove('%s' % out_file_path + '/' + '%s' % name + '_con.dbf')
    os.remove('%s' % out_file_path + '/' + '%s' % name + '_point.dbf')

    number_point = len(all_cp)

    file_all_cp = open('%s' % out_file_path + '/' + '%s' % name + '_point.txt', 'w')
    for p in range(0, number_point):
        file_all_cp.write('%d' % (all_cp[p][0] - 1))
        file_all_cp.write('% f' % all_cp[p][1])
        file_all_cp.write('% f' % all_cp[p][2])
        file_all_cp.write('% f\n' % all_cp[p][3])
    file_all_cp.close()


def read_point_data(name, in_file_path, out_file_path):
    f = open('%s' % in_file_path + '/' + '%s' % name + '.txt')
    point_coordinate = []
    for line in f:
        point_coordinate.append([float(x) for x in line.split()])
    f.close()

    num_point_total = len(point_coordinate)
    ID_elevation = []

    if num_point_total > 10:

        flag_find_ID = 0

        pre_ID = -1

        file_ID_elevation = open('%s' % in_file_path + '/ID_' + '%s' % name + '.txt', 'w')

        new_ID = 0

        for i in range(0, num_point_total):

            temp_ID = int(point_coordinate[i][0])
            temp_ele = point_coordinate[i][1]

            if temp_ID != pre_ID:
                flag = 0
                for j in range(0, 4):
                    if i + j >= num_point_total:
                        flag = 1
                    elif point_coordinate[i + j][0] != temp_ID:
                        flag = 1
                if flag == 0:
                    flag_find_ID = 1
                    # when the ID is changed, open a new file
                    file_contour = open('%s' % out_file_path + '/' + '%s' % name + '_' + str(int(new_ID)) + '.txt', 'w')

                    file_ID_elevation.write('%d' % new_ID)
                    file_ID_elevation.write(' %f' % temp_ele)
                    file_ID_elevation.write(' %d\n' % temp_ID)
                    ID_elevation.append([new_ID, temp_ele])

                    new_ID = new_ID + 1

            if flag == 0:
                file_contour.write('%f' % point_coordinate[i][2])
                file_contour.write(' %f\n' % point_coordinate[i][3])

                pre_ID = temp_ID
        if flag_find_ID == 1:
            file_contour.close()
        file_ID_elevation.close()
    else:
        new_ID = 0

    return new_ID, ID_elevation  # This is equal to number of IDs


###############################################################################
#######################   Function: poly_fit   ###############################
###############################################################################

def poly_fit(ID, name, in_file_path, out_file_path, number_of_uni_point, smoothness, x_0, y_0):
    ####################READING#####################################
    with open('%s' % in_file_path + '/' + '%s' % name + '_' + str(int(ID)) + '.txt') as f:
        temp_x = []
        temp_y = []
        temp_t = []
        len_contour = 0
        i = 0
        for line in f:
            temp_data = [float(temp_xx) for temp_xx in line.split()]
            temp_x.append(temp_data[0])
            temp_y.append(temp_data[1])
            if i > 0:
                temp_t.append(
                    temp_t[i - 1] + np.sqrt((temp_x[i] - temp_x[i - 1]) ** 2 + (temp_y[i] - temp_y[i - 1]) ** 2))
            else:
                temp_t.append(0)
            i = i + 1
    f.close()
    number_of_point = len(temp_x)

    os.remove('%s' % in_file_path + '/' + '%s' % name + '_' + str(int(ID)) + '.txt')
    # fix the direction of curve
    dis_1 = ((temp_x[0] - x_0) ** 2 + (temp_y[0] - y_0) ** 2) ** 0.5
    dis_2 = ((temp_x[number_of_point - 1] - x_0) ** 2 + (temp_y[number_of_point - 1] - y_0) ** 2) ** 0.5

    x = [0] * number_of_point
    y = [0] * number_of_point
    t = [0] * number_of_point

    if dis_2 < dis_1:
        for i in range(0, number_of_point):
            x[i] = temp_x[number_of_point - i - 1]
            y[i] = temp_y[number_of_point - i - 1]
            if i > 0:
                t[i] = temp_t[i - 1] + np.sqrt((temp_x[i] - temp_x[i - 1]) ** 2 + (temp_y[i] - temp_y[i - 1]) ** 2)
            else:
                t[i] = 0
    else:
        for i in range(0, number_of_point):
            x[i] = temp_x[i]
            y[i] = temp_y[i]
            t[i] = temp_t[i]

    x_0 = x[0]
    y_0 = y[0]

    ################################################################

    # Step 1: To fit a spilne to the contour line
    k = min(3, (len(x) - 1))  # spline order
    nest = -1  # estimate of number of knots needed (-1 = maximal)
    tckp, u = splprep([x, y, t], s=smoothness, k=k, nest=-1)

    x_new, y_new, t_new = splev(linspace(0, 1, number_of_uni_point), tckp)

    file_p_contour = open('%s' % out_file_path + '/' + '%s' % name + '_' + str(int(ID)) + '.txt', 'w')
    file_p_contour.write('ID rank x y kapa\n')

    file_p_contour.write('%d' % ID)
    file_p_contour.write('% d' % 0)
    file_p_contour.write('% f' % x_new[0])
    file_p_contour.write('% f ' % y_new[0])
    file_p_contour.write('% +f\n' % 0)

    dre_x_t = [0] * number_of_uni_point
    dre_y_t = [0] * number_of_uni_point
    dre_2_x_t = [0] * number_of_uni_point
    dre_2_y_t = [0] * number_of_uni_point
    kapa = [0] * number_of_uni_point

    for i in range(1, number_of_uni_point - 1):
        delta_t = t_new[i] - t_new[i - 1]

        dre_x_t[i] = (x_new[i] - x_new[i - 1]) / delta_t
        dre_y_t[i] = (y_new[i] - y_new[i - 1]) / delta_t

        dre_2_x_t[i] = (x_new[i + 1] - 2 * x_new[i] + x_new[i - 1]) / (delta_t) ** 2
        dre_2_y_t[i] = (y_new[i + 1] - 2 * y_new[i] + y_new[i - 1]) / (delta_t) ** 2

        kapa[i] = (dre_x_t[i] * dre_2_y_t[i] - dre_y_t[i] * dre_2_x_t[i]) / (dre_x_t[i] ** 2 + dre_y_t[i] ** 2) ** (
                3 / 2)

        file_p_contour.write('%d' % ID)
        file_p_contour.write('% d' % i)
        file_p_contour.write('% f' % x_new[i])
        file_p_contour.write('% f ' % y_new[i])
        file_p_contour.write('% +f\n' % kapa[i])

    file_p_contour.write('%d' % ID)
    file_p_contour.write('% d' % (number_of_uni_point - 1))
    file_p_contour.write('% f' % x_new[number_of_uni_point - 1])
    file_p_contour.write('% f ' % y_new[number_of_uni_point - 1])
    file_p_contour.write('% +f\n' % 0)

    file_p_contour.close()

    return kapa, x_0, y_0, x_new[int(number_of_uni_point / 2)], y_new[int(number_of_uni_point / 2)], x_new, y_new


#########################################################################################################################
def curvature(x, y, number_of_uni_point):
    dre_x_t = np.zeros((1, number_of_uni_point))
    dre_y_t = np.zeros((1, number_of_uni_point))
    dre_2_x_t = np.zeros((1, number_of_uni_point))
    dre_2_y_t = np.zeros((1, number_of_uni_point))
    kapa = np.zeros((1, number_of_uni_point))

    for i in range(1, number_of_uni_point - 1):
        delta_t = ((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) ** 0.5
        dre_x_t[0, i] = (x[i] - x[i - 1]) / delta_t
        dre_y_t[0, i] = (y[i] - y[i - 1]) / delta_t

        dre_2_x_t[0, i] = (x[i + 1] - 2 * x[i] + x[i - 1]) / (delta_t) ** 2
        dre_2_y_t[0, i] = (y[i + 1] - 2 * y[i] + y[i - 1]) / (delta_t) ** 2

        kapa[0, i] = (dre_x_t[0, i] * dre_2_y_t[0, i] - dre_y_t[0, i] * dre_2_x_t[0, i]) / (
                dre_x_t[0, i] ** 2 + dre_y_t[0, i] ** 2) ** (3 / 2)

    return np.max(np.abs(kapa))


#######################################################################################################
#######################################################################################################

def plot_contour(ID, text_file_path):
    f = open('%s' % text_file_path + '/p_contour_' + str(int(ID)) + '.txt')
    contour = []
    next(f)
    i = 0
    for line in f:
        contour.append([float(x) for x in line.split()])
        if contour[i][4] <= 0:
            pl.plot(contour[i][2], contour[i][3], 'r+', linewidth=0.1)
        else:
            pl.plot(contour[i][2], contour[i][3], 'g*', linewidth=0.1)
        i = i + 1
    f.close()


#######################################################################################################
#######################################################################################################

def whole_to_part(kapa_1, kapa_2):
    number_of_point_1 = len(kapa_1)
    number_of_point_2 = len(kapa_2)

    if number_of_point_2 >= number_of_point_1:

        ave_1 = 0
        for i in range(1, number_of_point_1):
            ave_1 = ave_1 + kapa_1[i] / number_of_point_1

        max_u = number_of_point_2 - number_of_point_1 + 1
        v = [0] * max_u
        max_v = -5
        best_u = 0
        for u in range(0, max_u):
            ave_2 = 0
            for i in range(0, number_of_point_1):
                ave_2 = ave_2 + kapa_2[i + u] / number_of_point_1
                if i == int(number_of_point_1 / 2):
                    max_kapa_2 = abs(kapa_2[i + u])
            naminator = 0
            denaminator_1 = 0
            denaminator_2 = 0
            for i in range(0, number_of_point_1):
                naminator = naminator + (kapa_1[i] - ave_1) * (kapa_2[i + u] - ave_2)
                denaminator_1 = denaminator_1 + (kapa_1[i] - ave_1) ** 2
                denaminator_2 = denaminator_2 + (kapa_2[i + u] - ave_2) ** 2
            if (denaminator_1 * denaminator_2) == 0:
                v[u] = -5
            else:
                v[u] = naminator / (denaminator_1 * denaminator_2) ** 0.5

            if v[u] > max_v:
                max_v = v[u]
                best_u = u

    else:
        max_v = -5

    return max_v


#######################################################################################################
#######################################################################################################

def curve_modify(kapa):
    number_of_point = len(kapa)
    mid_point = int(number_of_point / 2)
    max_kapa = 0
    mid_i = mid_point
    for i in range(0, number_of_point):
        if kapa[i] < max_kapa:
            max_kapa = kapa[i]
            mid_i = i
    x = [0] * number_of_point
    for i in range(0, number_of_point):
        x[i] = float(i) / mid_i * mid_point
    for i in range(mid_point + 1, number_of_point):
        x[i] = x[i] / x[number_of_point - 1] * number_of_point

    kapa_new = [0] * number_of_point
    for j in range(0, number_of_point):
        for i in range(0, number_of_point - 1):
            if j >= x[i] and j <= x[i + 1]:
                kapa_new[j] = kapa[i] + (kapa[i + 1] - kapa[i]) / (x[i + 1] - x[i]) * (j - x[i])

    return kapa_new


####################################################################################################

####################################################################################################

def contour_curve(x, y, curve_Array, number_of_point, number_of_ID, X_min, Y_max, dx, dy):
    c = np.zeros_like(x)
    for ID in range(0, number_of_ID):
        for i in range(0, number_of_point):
            ii = int((Y_max - dy / 2 - y[ID, i]) / dy)
            jj = int((x[ID, i] - X_min - dx / 2) / dx)
            c[ID, i] = curve_Array[ii, jj]
    return c


def contour_sym_curve(x, y, org_x, org_y, curve_Array, basin_Array, ridge_Array, number_of_point, number_of_ID, X_min,
                      Y_max, dx, dy):
    sym_x = np.zeros_like(x)
    sym_y = np.zeros_like(y)

    mid_i = np.zeros((1, number_of_ID))

    for ID in range(0, number_of_ID):
        max_curve = 0
        for i in range(0, number_of_point):
            ii = int((Y_max - dy / 2 - org_y[ID, i]) / dy)
            jj = int((org_x[ID, i] - X_min - dx / 2) / dx)
            if curve_Array[ii, jj] > max_curve and basin_Array[ii, jj] == 0:
                max_curve = curve_Array[ii, jj]
                mid_i[0, ID] = i
        print
        mid_i[0, ID], max_curve

    mid_i = mid_i.astype(int)
    for ID in range(0, number_of_ID):
        upper_bond = number_of_point - 1
        for i in range(mid_i[0, ID], number_of_point - 3):
            ii = int((Y_max - dy / 2 - org_y[ID, i]) / dy)
            jj = int((org_x[ID, i] - X_min - dx / 2) / dx)

            flag_end_ridge = 1
            for next_i in range(i + 1, i + 3):
                next_ii = int((Y_max - dy / 2 - org_y[ID, next_i]) / dy)
                next_jj = int((org_x[ID, next_i] - X_min - dx / 2) / dx)
                if ridge_Array[next_ii, next_jj] == 1:
                    flag_end_ridge = 0
            if ridge_Array[ii, jj] == 1 and flag_end_ridge == 1:
                upper_bond = i
                break

        lower_bond = 0
        for i in range(mid_i[0, ID], 3, -1):
            ii = int((Y_max - dy / 2 - org_y[ID, i]) / dy)
            jj = int((org_x[ID, i] - X_min - dx / 2) / dx)

            flag_end_ridge = 1
            for next_i in range(i - 1, i - 3, -1):
                next_ii = int((Y_max - dy / 2 - org_y[ID, next_i]) / dy)
                next_jj = int((org_x[ID, next_i] - X_min - dx / 2) / dx)
                if ridge_Array[next_ii, next_jj] == 1:
                    flag_end_ridge = 0
            if ridge_Array[ii, jj] == 1 and flag_end_ridge == 1:
                lower_bond = i
                break

        new_x = np.zeros((1, 1))
        new_y = np.zeros((1, 1))
        for i in range(lower_bond, upper_bond):
            new_x = np.copy(np.append(new_x, x[ID, i].reshape(1, 1)))
            new_y = np.copy(np.append(new_y, y[ID, i].reshape(1, 1)))
        new_x = np.delete(new_x, 0, 0)
        new_y = np.delete(new_y, 0, 0)
        new_x_1, new_y_1 = simple_poly_fit(new_x, new_y, number_of_point, 0.1)
        sym_x[ID, :] = new_x_1
        sym_y[ID, :] = new_y_1

    return sym_x, sym_y


def contour_sym(x, y, number_of_point, number_of_ID):
    sym_x = np.zeros_like(x)
    sym_y = np.zeros_like(y)

    for ID in range(0, number_of_ID):
        max_y = 0
        for i in range(0, number_of_point):
            if -1 * (y[ID, i]) > max_y:
                max_y = -1 * y[ID, i]
                mid_i = i
        # print x[ID , :]
        # print y[ID , :]
        # print mid_i , max_y
        new_x = np.zeros((1, 1))
        new_y = np.zeros((1, 1))
        if mid_i <= number_of_point / 2:
            for i in range(0, mid_i + 1):
                new_x = np.copy(np.append(new_x, x[ID, i].reshape(1, 1)))
                new_y = np.copy(np.append(new_y, y[ID, i].reshape(1, 1)))
            new_x = np.delete(new_x, 0, 0)
            new_y = np.delete(new_y, 0, 0)
            # print new_x
            pre_x = new_x[mid_i]
            for i in range(0, mid_i):
                pre_x = pre_x + (x[ID, mid_i - i] - x[ID, mid_i - i - 1])
                new_x = np.copy(np.append(new_x, pre_x * np.ones((1, 1))))
                new_y = np.copy(np.append(new_y, y[ID, mid_i - i - 1].reshape(1, 1)))
        else:

            pre_x = x[ID, number_of_point - 1] - 2 * (x[ID, number_of_point - 1] - x[ID, mid_i])
            for i in range(0, number_of_point - mid_i - 1):
                new_x = np.copy(np.append(new_x, pre_x * np.ones((1, 1))))
                new_y = np.copy(np.append(new_y, y[ID, number_of_point - i - 1].reshape(1, 1)))
                pre_x = pre_x + x[ID, number_of_point - i - 1] - x[ID, number_of_point - i - 2]

            new_x = np.delete(new_x, 0, 0)
            new_y = np.delete(new_y, 0, 0)
            for i in range(0, number_of_point - mid_i):
                new_x = np.copy(np.append(new_x, pre_x * np.ones((1, 1))))
                new_y = np.copy(np.append(new_y, y[ID, mid_i + i].reshape(1, 1)))
                if i < (number_of_point - mid_i - 1):
                    pre_x = pre_x + (x[ID, mid_i + i + 1] - x[ID, mid_i + i])

        # pl.plot(new_x , new_y , '-b')
        # pl.plot(x[ID , :] , y[ID , :] , '-r')

        new_x_1, new_y_1 = simple_poly_fit(new_x, new_y, number_of_point, 1)
        # new_x_1 = new_x_1 - new_x_1[0]
        # new_y_1 = new_y_1 - new_y_1[0]

        # pl.plot(new_x_1 , new_y_1 , 'og')

        # print new_x
        # print new_y
        sym_x[ID, :] = new_x_1
        sym_y[ID, :] = new_y_1

    return sym_x, sym_y


############################################################################

def simple_poly_fit(x, y, number_of_uni_point, smoothness):
    number_of_point = len(x)
    t = np.zeros_like(x)
    for i in range(0, number_of_point):
        if i > 0:
            t[i] = t[i - 1] + np.sqrt((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2)
        else:
            t[i] = 0

    k = min(3, number_of_point - 1)  # spline order
    nest = -1  # estimate of number of knots needed (-1 = maximal)
    tckp, u = splprep([x, y, t], s=smoothness, k=k, nest=-1)

    x_new, y_new, t_new = splev(linspace(0, 1, number_of_uni_point), tckp)

    return x_new, y_new


########################################################################
########################################################################


def next_contour(ID, text_file_path, number_of_ID, contour_interval):
    f = open('%s' % text_file_path + '/ID_elevation.txt')
    ID_elevation = []
    for line in f:
        ID_elevation.append([float(temp_x) for temp_x in line.split()])
    f.close()

    ele = ID_elevation[ID][1]

    point = []
    f = open('%s' % text_file_path + '/p_contour_' + str(int(ID)) + '.txt')
    next(f)
    for line in f:
        point.append([float(temp_x) for temp_x in line.split()])
    f.close()

    number_point = len(point)

    index_s = int(0.05 * number_point)
    index_e = int(0.95 * number_point) - 1

    x_s = point[index_s][2]
    y_s = point[index_s][3]

    x_e = point[index_e][2]
    y_e = point[index_e][3]

    min_dis_s = 10 ** 5
    min_dis_e = 10 ** 5
    next_ID_s = - 1
    next_ID_e = - 1
    scale = -1

    point_other = []
    j = 0
    for other_ID in range(0, number_of_ID):
        if abs(ID_elevation[other_ID][1] - ele) <= contour_interval and ele < ID_elevation[other_ID][1]:
            f = open('%s' % text_file_path + '/p_contour_' + str(int(other_ID)) + '.txt')
            next(f)
            for line in f:
                point_other.append([float(temp_x) for temp_x in line.split()])
                distance_s = ((x_s - point_other[j][2]) ** 2 + (y_s - point_other[j][3]) ** 2) ** 0.5
                distance_e = ((x_e - point_other[j][2]) ** 2 + (y_e - point_other[j][3]) ** 2) ** 0.5
                if distance_s < min_dis_s:
                    next_ID_s = point_other[j][0]
                    min_dis_s = distance_s
                    j_s = j

                if distance_e < min_dis_e:
                    next_ID_e = point_other[j][0]
                    min_dis_e = distance_e
                    j_e = j

                j = j + 1
            f.close()
    if min_dis_s < 100 * contour_interval and min_dis_e < 100 * contour_interval and next_ID_e == next_ID_s:
        scale = abs((j_e - j_s) / (index_e - min_dis_s))
        for i in range(index_s, index_e + 1):
            pl.plot(point[i][2], point[i][3], '.g', linewidth=0.1)

        for i in range(j_s, j_e + 1):
            pl.plot(point_other[i][2], point_other[i][3], '.r', linewidth=0.1)

    return scale


###########################################################################################

def best_orientation(A, B):
    number_of_point = A.shape[1]

    ave_A = np.zeros((2, 1))
    ave_B = np.zeros((2, 1))

    for i in range(0, number_of_point):
        ave_A[0] = ave_A[0] + A[0][i] / number_of_point
        ave_A[1] = ave_A[1] + A[1][i] / number_of_point

        ave_B[0] = ave_B[0] + B[0][i] / number_of_point
        ave_B[1] = ave_B[1] + B[1][i] / number_of_point

    A_cen = np.zeros((2, number_of_point))
    B_cen = np.zeros((2, number_of_point))

    for i in range(0, number_of_point):
        A_cen[0][i] = A[0][i] - ave_A[0]
        A_cen[1][i] = A[1][i] - ave_A[1]

        B_cen[0][i] = B[0][i] - ave_B[0]
        B_cen[1][i] = B[1][i] - ave_B[1]

    M = np.zeros((2, 2))

    for j in range(0, 2):
        for k in range(0, 2):
            for i in range(0, number_of_point):
                M[j][k] = M[j][k] + A_cen[j][i] * B_cen[k][i]

    N_xx = M[0][0] + M[1][1]
    N_yx = M[0][1] - M[1][0]

    N = np.array([[N_xx, N_yx], [N_yx, -N_xx]])

    D, V = np.linalg.eig(N)

    if D[0] >= D[1]:
        e_max = 0
    else:
        e_max = 1

    q = np.array([[V[0][e_max]], [V[1][e_max]]])

    if q[1][0] + (q[1][0] >= 0) >= 0:
        q = 1 * q
    else:
        q = -1 * q

    q = np.transpose(q) / np.linalg.norm(q)

    R_11 = q[0][0] ** 2 - q[0][1] ** 2
    R_21 = 2 * q[0][0] * q[0][1]

    R = np.array([[R_11, -R_21], [R_21, R_11]])

    # print R

    temp_1 = np.zeros((2, number_of_point))
    temp_2 = np.zeros((2, number_of_point))

    temp_sum_1 = 0
    temp_sum_2 = 0

    for j in range(0, 2):
        for i in range(0, number_of_point):
            for k in range(0, 2):
                temp_1[j][i] = temp_1[j][i] + A_cen[k][i] * R[j][k]

            temp_2[j][i] = temp_1[j][i] * B_cen[j][i]
            temp_sum_1 = temp_sum_1 + temp_2[j][i]

            temp_sum_2 = temp_sum_2 + A_cen[j][i] ** 2

    scale = temp_sum_1 / temp_sum_2

    # scale = 1

    # print scale

    temp_t = np.zeros((2, 1))
    t = np.zeros((2, 1))

    for j in range(0, 2):
        for k in range(0, 2):
            temp_t[j][0] = temp_t[j][0] + R[j][k] * ave_A[k][0] * scale
        t[j][0] = ave_B[j][0] - temp_t[j][0]

    # print t

    new_A = np.zeros((2, number_of_point))

    for j in range(0, 2):
        for i in range(0, number_of_point):
            for k in range(0, 2):
                new_A[j][i] = new_A[j][i] + R[j][k] * A[k][i] * scale

            new_A[j][i] = new_A[j][i] + t[j][0]

    R_sqe = 0
    for i in range(0, number_of_point):
        R_sqe = R_sqe + ((new_A[0][i] - B[0][i]) ** 2 + (new_A[1][i] - B[1][i]) ** 2)

    R_sqe = R_sqe ** 0.5

    return R_sqe


####################################################################################
####################################################################################

def contour_direction_fix(X, Y, ID_elevation, number_of_ID, number_of_uni_point):
    mid_index = int(number_of_uni_point / 2)
    ## X , Y  = rotation(X , Y ,  number_of_uni_point , number_of_ID)
    for ID in range(0, number_of_ID - 1):
        next_ID = ID + 1

        V_1 = [(X[next_ID, mid_index] - X[ID, mid_index]), (Y[next_ID, mid_index] - Y[ID, mid_index])]
        V_2 = [(X[ID, mid_index - 1] - X[ID, mid_index]), (Y[ID, mid_index - 1] - Y[ID, mid_index])]
        V_3 = [(X[ID, number_of_uni_point - 1] - X[next_ID, mid_index]),
               (Y[ID, number_of_uni_point - 1] - Y[next_ID, mid_index])]

        if np.cross(V_1, V_2) < 0:
            temp_X = np.copy(X[ID, :])
            temp_Y = np.copy(Y[ID, :])
            for i in range(0, number_of_uni_point):
                X[ID][i] = temp_X[number_of_uni_point - i - 1]
                Y[ID][i] = temp_Y[number_of_uni_point - i - 1]

    ID = number_of_ID - 1
    next_ID = ID - 1

    V_1 = [(X[next_ID, mid_index] - X[ID, mid_index]), (Y[next_ID, mid_index] - Y[ID, mid_index])]
    V_2 = [(X[ID, mid_index - 1] - X[ID, mid_index]), (Y[ID, mid_index - 1] - Y[ID, mid_index])]

    if np.cross(V_1, V_2) > 0:
        temp_X = np.copy(X[ID, :])
        temp_Y = np.copy(Y[ID, :])
        for i in range(0, number_of_uni_point):
            X[ID][i] = temp_X[number_of_uni_point - i - 1]
            Y[ID][i] = temp_Y[number_of_uni_point - i - 1]
    return X, Y


####################################################################################
####################################################################################

def rotation(X, Y, number_of_uni_point, number_of_ID):
    new_X = np.zeros_like(X)
    new_Y = np.zeros_like(Y)
    for ID in range(0, number_of_ID):
        Teta = -1 * math.atan2((Y[ID, number_of_uni_point - 1] - Y[ID, 0]), (X[ID, number_of_uni_point - 1] - X[ID, 0]))
        for i in range(0, number_of_uni_point):
            new_X[ID, i] = (X[ID, i]) * math.cos(Teta) - (Y[ID, i]) * math.sin(Teta)
            new_Y[ID, i] = (X[ID, i]) * math.sin(Teta) + (Y[ID, i]) * math.cos(Teta)

    return new_X, new_Y


##############################################################################
##############################################################################
def inv_rotation(org_X, org_Y, X, Y, number_of_uni_point, number_of_ID):
    new_X = np.zeros_like(X)
    new_Y = np.zeros_like(Y)
    for ID in range(0, number_of_ID):
        Teta = math.atan2((org_Y[ID, number_of_uni_point - 1] - org_Y[ID, 0]),
                          (org_X[ID, number_of_uni_point - 1] - org_X[ID, 0]))
        for i in range(0, number_of_uni_point):
            new_X[ID, i] = (X[ID, i]) * math.cos(Teta) - (Y[ID, i]) * math.sin(Teta) + org_X[ID, 0]
            new_Y[ID, i] = (X[ID, i]) * math.sin(Teta) + (Y[ID, i]) * math.cos(Teta) + org_Y[ID, 0]

    return new_X, new_Y


####################################################################################
####################################################################################
def contour_delete(X, Y, ID_elevation, number_of_ID, number_of_uni_point, contour_interval, max_ele, min_ele):
    ##############################################################
    sum_deleted = 0
    deleted_contour = np.zeros((number_of_ID, 1))
    for ID in range(0, number_of_ID):
        if ID_elevation[ID][1] > max_ele or ID_elevation[ID][1] < min_ele:
            deleted_contour[ID][0] = 1
            sum_deleted = sum_deleted + 1

    new_0_number_of_ID = int(number_of_ID - sum_deleted)

    new_0_X = np.zeros((new_0_number_of_ID, number_of_uni_point))
    new_0_Y = np.zeros((new_0_number_of_ID, number_of_uni_point))
    new_0_ID_elevation = np.zeros((new_0_number_of_ID, 2))
    c_ID = 0
    for ID in range(0, number_of_ID):
        if deleted_contour[ID][0] == 0:
            new_0_ID_elevation[c_ID][0] = ID_elevation[ID][0]
            new_0_ID_elevation[c_ID][1] = ID_elevation[ID][1]
            for i in range(0, number_of_uni_point):
                new_0_X[c_ID][i] = X[ID][i]
                new_0_Y[c_ID][i] = Y[ID][i]

            c_ID = c_ID + 1

    average_len = 0
    contour_len = np.zeros((new_0_number_of_ID, 1))
    for ID in range(0, new_0_number_of_ID):
        for i in range(0, number_of_uni_point - 1):
            contour_len[ID][0] = contour_len[ID][0] + (
                    (new_0_X[ID][i] - new_0_X[ID][i + 1]) ** 2 + (new_0_Y[ID][i] - new_0_Y[ID][i + 1]) ** 2) ** 0.5
        average_len = average_len + contour_len[ID][0] / new_0_number_of_ID

    deleted_code = np.zeros((1, new_0_number_of_ID))
    num_delete = 1
    while num_delete > 0:
        num_delete = 0
        for ID_1 in range(0, new_0_number_of_ID):
            if contour_len[ID_1] < 0.1 * average_len and deleted_code[0][ID_1] == 0:
                deleted_code[0][ID_1] = 1
                num_delete = num_delete + 1

            for ID_2 in range(ID_1 + 1, new_0_number_of_ID):
                if deleted_code[0][ID_2] == 0 and deleted_code[0][ID_1] == 0:
                    if contour_len[ID_1][0] < 0.8 * contour_len[ID_2][0] or new_0_ID_elevation[ID_1][0] == \
                            new_0_ID_elevation[ID_2][0] or contour_len[ID_1] < 0.1 * average_len:
                        deleted_code[0][ID_1] = 1
                        num_delete = num_delete + 1
                    break
    sum_deleted_code = 0
    for ID in range(0, new_0_number_of_ID):
        sum_deleted_code = sum_deleted_code + deleted_code[0][ID]

    new_1_number_of_ID = int(new_0_number_of_ID - sum_deleted_code)

    new_1_X = np.zeros((new_1_number_of_ID, number_of_uni_point))
    new_1_Y = np.zeros((new_1_number_of_ID, number_of_uni_point))
    new_1_ID_elevation = np.zeros((new_1_number_of_ID, 2))
    c_ID = 0
    for ID in range(0, new_0_number_of_ID):
        if deleted_code[0][ID] == 0:
            new_1_ID_elevation[c_ID][0] = new_0_ID_elevation[ID][0]
            new_1_ID_elevation[c_ID][1] = new_0_ID_elevation[ID][1]
            for i in range(0, number_of_uni_point):
                new_1_X[c_ID][i] = new_0_X[ID][i]
                new_1_Y[c_ID][i] = new_0_Y[ID][i]

            c_ID = c_ID + 1

    return new_1_X, new_1_Y, new_1_ID_elevation, new_1_number_of_ID


########################################################################################
########################################################################################

def cluster_assign(x_data, y_data, x_ave, y_ave, number_of_k, number_of_ID, number_of_uni_point, old_cluster_code,
                   min_number_contour):
    old_count_code = np.zeros((1, number_of_k))
    for k in range(0, number_of_k):
        for ID in range(0, number_of_ID):
            if old_cluster_code[0][ID] == k:
                old_count_code[0][k] = old_count_code[0][k] + 1

    cluster_code = np.copy(old_cluster_code)
    count_code = np.copy(old_count_code)

    flag_change = 0
    for ID in range(1, number_of_ID - 1):
        old_k = int(old_cluster_code[0][ID])
        if count_code[0][old_k] > min_number_contour[old_k]:
            if old_cluster_code[0][ID] != old_cluster_code[0][ID + 1] or old_cluster_code[0][ID] != old_cluster_code[0][
                ID - 1] and flag_change == 0:
                min_diff = 10 ** 10

                A = np.zeros((2, number_of_uni_point))
                for i in range(0, number_of_uni_point):
                    A[0][i] = x_data[ID][i]
                    A[1][i] = y_data[ID][i]

                for k in range(0, number_of_k):
                    B = np.zeros((2, number_of_uni_point))
                    for i in range(0, number_of_uni_point):
                        B[0][i] = x_ave[k][i]
                        B[1][i] = y_ave[k][i]

                    diff = best_orientation(B, A)
                    if diff < min_diff:
                        min_diff = diff
                        best_k = k
                        flag_add = 1
                cluster_code[0][ID] = best_k

                if best_k != old_k:
                    count_code[0][best_k] = count_code[0][best_k] + 1
                    count_code[0][old_k] = count_code[0][old_k] - 1
                    flag_change = 1

    return cluster_code


def cluster_average(cluster_code, x_data, y_data, number_of_k, number_of_ID, number_of_uni_point):
    x_ave = np.zeros((number_of_k, number_of_uni_point))
    y_ave = np.zeros((number_of_k, number_of_uni_point))

    for k in range(0, number_of_k):
        count_k = 0
        for ID in range(0, number_of_ID):
            if cluster_code[0][ID] == k:
                for i in range(0, number_of_uni_point):
                    x_ave[k][i] = x_ave[k][i] + x_data[ID][i]
                    y_ave[k][i] = y_ave[k][i] + y_data[ID][i]
                count_k = count_k + 1
        for i in range(0, number_of_uni_point):
            x_ave[k][i] = x_ave[k][i] / float(count_k)
            y_ave[k][i] = y_ave[k][i] / float(count_k)
    return x_ave, y_ave


def cluster_performance(x_data, y_data, cluster_code, x_ave, y_ave, number_of_k, number_of_ID, number_of_uni_point):
    perf = 0
    A = np.zeros((2, number_of_uni_point))
    B = np.zeros((2, number_of_uni_point))

    for k in range(0, number_of_k):
        for i in range(0, number_of_uni_point):
            B[0][i] = x_ave[k][i]
            B[1][i] = y_ave[k][i]

        for ID in range(0, number_of_ID):
            if cluster_code[0][ID] == k:
                for i in range(0, number_of_uni_point):
                    A[0][i] = x_data[ID][i]
                    A[1][i] = y_data[ID][i]

                perf = perf + best_orientation(B, A)

    return perf


def cluster_find(x_data, y_data, number_of_k, number_of_ID, number_of_uni_point, min_contour, initial_cluster_code):
    min_number_contour = [min_contour, min_contour]

    cluster_code = np.copy(initial_cluster_code)

    old_cluster_code = np.random.randint(2, size=(1, number_of_ID))

    x_ave, y_ave = cluster_average(cluster_code, x_data, y_data, number_of_k, number_of_ID, number_of_uni_point)

    flag = 1
    trail = 0

    while flag == 1:
        trail = trail + 1
        old_cluster_code = cluster_code

        # Assign all points in X to clusters
        cluster_code = cluster_assign(x_data, y_data, x_ave, y_ave, number_of_k, number_of_ID, number_of_uni_point,
                                      old_cluster_code, min_number_contour)

        # print trail , cluster_code , elevation_bond

        # Reevaluate centers
        x_ave, y_ave = cluster_average(cluster_code, x_data, y_data, number_of_k, number_of_ID, number_of_uni_point)

        flag = 0
        for ID in range(0, number_of_ID):
            if cluster_code[0][ID] != old_cluster_code[0][ID]:
                flag = 1

        if trail >= 100:
            flag = 0

    return cluster_code, x_ave, y_ave, trail
