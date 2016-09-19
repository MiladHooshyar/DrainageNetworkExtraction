import arcpy
from arcpy import env
from arcpy.sa import *
import numpy as np
import scipy
from scipy import ndimage, sparse
from scipy.stats import norm
import itertools
import math
import os
import matplotlib.pyplot as pl

arcpy.CheckOutExtension("Spatial")


def segment_size_curvature(curve_Array):
    min_c = np.nanmin(curve_Array)
    max_c = np.nanmax(curve_Array)
    max_c = max(max_c , -min_c)
    
    max_num_label_valley = 0
    max_num_label_ridge = 0
    
    d_curve = max_c / (5 * 100)
    for i in range (0 , 100):
        k_tresh_valley = 1 * ( float(i) * d_curve)
        k_tresh_ridge = -1 * k_tresh_valley
        
        Val_Array = np.where(curve_Array >= k_tresh_valley, 1, 0)
        Rid_Array = np.where(curve_Array <= k_tresh_ridge, 1, 0)
    
        Lab_Val_Array, num_label_val = ndimage.label(Val_Array , structure = np.ones((3 , 3)))
        Lab_Rid_Array, num_label_rid = ndimage.label(Rid_Array , structure = np.ones((3 , 3)))
    
        labels_val = np.arange(1, num_label_val + 1)
        labels_rid = np.arange(1, num_label_rid + 1)

        try:
            area_label_val = ndimage.labeled_comprehension(Val_Array, Lab_Val_Array, labels_val, np.sum, int, 0)
            num_sig_label_valley = np.sum(area_label_val >= 1)
        except ValueError:
            num_sig_label_valley = 0
            
        try:
            area_label_rid = ndimage.labeled_comprehension(Rid_Array, Lab_Rid_Array, labels_rid, np.sum, int, 0)
            num_sig_label_ridge = np.sum(area_label_rid >= 1)
        except ValueError:
            num_sig_label_ridge = 0

        
        if num_sig_label_valley >= max_num_label_valley:
            max_num_label_valley = num_sig_label_valley
            valley_thresh = k_tresh_valley
        if num_sig_label_ridge >= max_num_label_ridge:
            max_num_label_ridge = num_sig_label_ridge
            ridge_thresh = k_tresh_ridge
                
    return valley_thresh , ridge_thresh

def neg_bond(curvature_file, valley_bond_file , ridge_bond_file , covergent_sig_file , divergent_sig_file):
    
    curve_Array = arcpy.RasterToNumPyArray(curvature_file  , nodata_to_value=0)
    corner = arcpy.Point(arcpy.Describe(curvature_file).Extent.XMin,arcpy.Describe(curvature_file).Extent.YMin)
    dx = arcpy.Describe(curvature_file).meanCellWidth

    valley_thresh , ridge_thresh = segment_size_curvature(curve_Array)


    valley = np.where(curve_Array > valley_thresh , 1 , 0)
    ridge = np.where(curve_Array < ridge_thresh , 1 , 0)
    
    Lab_Val_Array, num_label_val = ndimage.label(valley , structure = np.ones((3 , 3)))
    Lab_Rid_Array, num_label_rid = ndimage.label(ridge , structure = np.ones((3 , 3)))

    perc90_valley = ndimage.labeled_comprehension(curve_Array, Lab_Val_Array, np.arange(1, num_label_val + 1) , percentile, float, 0)
    perc90_ridge = ndimage.labeled_comprehension(-1 * curve_Array, Lab_Rid_Array, np.arange(1, num_label_rid + 1) , percentile, float, 0)

    area_valley = ndimage.labeled_comprehension(valley, Lab_Val_Array, np.arange(1, num_label_val + 1) , np.sum, float, 0)
    area_ridge = ndimage.labeled_comprehension(ridge, Lab_Rid_Array, np.arange(1, num_label_rid + 1) , np.sum, float, 0)

    main_patch_val = perc90_valley[area_valley == np.max(area_valley)][0]
    main_patch_ridge = perc90_ridge[area_ridge == np.max(area_ridge)][0]


    sig_valley_thresh = otsu(perc90_valley[perc90_valley < main_patch_val])
    sig_ridge_thresh = otsu(perc90_ridge[perc90_ridge < main_patch_ridge])

    perc90_valley = perc90_valley[Lab_Val_Array - 1]
    perc90_valley = np.where(valley == 1 , perc90_valley , 0)
    
    perc90_ridge = perc90_ridge[Lab_Rid_Array - 1]
    perc90_ridge = np.where(ridge == 1 , perc90_ridge , 0)

    sig_valley_patch = np.where(perc90_valley > sig_valley_thresh , 1 , 0)
    sig_ridge_patch = np.where(perc90_ridge > sig_ridge_thresh , 1 , 0)

    arcpy.NumPyArrayToRaster(perc90_valley , corner,dx,dx , value_to_nodata=0).save(covergent_sig_file)
    arcpy.NumPyArrayToRaster(perc90_ridge , corner,dx,dx , value_to_nodata=0).save(divergent_sig_file)
    
    arcpy.NumPyArrayToRaster(sig_valley_patch , corner,dx,dx , value_to_nodata=0).save(valley_bond_file)
    arcpy.NumPyArrayToRaster(sig_ridge_patch , corner,dx,dx , value_to_nodata=0).save(ridge_bond_file)

    return valley_thresh


def otsu(A):
    max_A = np.max(A)
    min_A = np.min(A)
    dA = (max_A - min_A) / 100

    max_Sig = 0
    for i in range (1 , 99):
        thresh_A = min_A + i * dA
        mu_1 = np.mean(A[A >= thresh_A])
        mu_2 = np.mean(A[A < thresh_A])

        w_1 = np.sum(A >= thresh_A)
        w_2 = np.sum(A < thresh_A)

        Sig = w_1 * w_2 * (mu_1 - mu_2) ** 2

##        print thresh_A , Sig

        if Sig > max_Sig:
            max_Sig = Sig
            T = thresh_A
    return T
        

def percentile(A):
    p = np.percentile(A , 95)
    return p

def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return average, math.sqrt(variance)

def kmeans_clustering(x , k):
    y = KMeans(n_clusters=2 , random_state = 200).fit_predict(x)
    pl.plot(x , y , 'o')
    pl.show()

def significant_patch(curve , figure_file):
    y=range(0 , 100 , 1)
    x = np.percentile(curve , y)
    x = np.asarray(x , dtype=np.float32)
    y = np.asarray(y , dtype=np.float32) / 100
    sig_thresh = general_linear_fit_lin(x , y , figure_file)
    return sig_thresh


def Perona_malik(DEM_file , filt_DEM_file , num_iter , resolution , option):
        dem_Array = arcpy.RasterToNumPyArray(DEM_file , nodata_to_value = 0)
        corner = arcpy.Point(arcpy.Describe(DEM_file).Extent.XMin,arcpy.Describe(DEM_file).Extent.YMin)
        dx = arcpy.Describe(DEM_file).meanCellWidth
        filt_dem_Array  = Perona_malik_iter(dem_Array,  num_iter , resolution , option)
        filt_dem_Array = np.where(dem_Array == 0, 0 , filt_dem_Array)
        arcpy.NumPyArrayToRaster(filt_dem_Array , corner , dx , dx , value_to_nodata = 0).save(filt_DEM_file)
        
def Perona_malik_iter(dem_Array, num_iter , resolution ,option):

        grad_x , grad_y =  np.gradient(dem_Array , resolution)
        abs_grad = np.sqrt(grad_x ** 2 + grad_y ** 2)
        extent = np.where(dem_Array > 0 , 1 , 0).astype(np.int8)
        extent = scipy.ndimage.binary_erosion(extent , iterations = 3).astype(extent.dtype)
        abs_grad = np.where(extent == 1, abs_grad , 0)
        abs_grad = abs_grad[abs_grad > 0]
        edge_thresh = 3 * np.percentile(abs_grad , 90)

        Ni =  dem_Array.shape[0]
        Nj =  dem_Array.shape[1]

        for it in range (0 , num_iter):
                
                if 0.01 > 0:
                        org_dem_Array = np.copy(dem_Array)
                        dem_Array = scipy.ndimage.filters.gaussian_filter(dem_Array , 0.01)
                
                Ni =  dem_Array.shape[0]
                Nj =  dem_Array.shape[1]
                
                # Calculate drevtives in for directions
                delta_n = (np.append(dem_Array[0 , :].reshape(1 , Nj) , dem_Array[0 : Ni - 1  , :].reshape(Ni - 1 , Nj) , axis = 0) - dem_Array) / resolution
                delta_s = (np.append(dem_Array[1 : Ni  , :].reshape(Ni - 1 , Nj) , dem_Array[Ni - 1 , :].reshape(1 , Nj) , axis = 0) - dem_Array) / resolution
                delta_e = (np.append(dem_Array[: , 1 : Nj].reshape(Ni , Nj - 1) , dem_Array[: , Nj - 1].reshape(Ni , 1) , axis = 1) - dem_Array) / resolution
                delta_w = (np.append(dem_Array[: , 0].reshape(Ni , 1) , dem_Array[: , 0 : Nj - 1].reshape(Ni , Nj - 1) , axis = 1) - dem_Array) / resolution
                delta_n = np.nan_to_num(delta_n)
                delta_s = np.nan_to_num(delta_s)
                delta_e = np.nan_to_num(delta_e)
                delta_w = np.nan_to_num(delta_w)

                if option == 'PM1':
                        Cn = np.exp(-(np.abs(delta_n)/edge_thresh)**2.)
                        Cs = np.exp(-(np.abs(delta_s)/edge_thresh)**2.)
                        Ce = np.exp(-(np.abs(delta_e)/edge_thresh)**2.)
                        Cw = np.exp(-(np.abs(delta_w)/edge_thresh)**2.)
                        del delta_n , delta_s , delta_e , delta_w
                elif option == 'PM2':
                        Cn = 1./(1.+(np.abs(delta_n)/edge_thresh)**2.)
                        Cs = 1./(1.+(np.abs(delta_s)/edge_thresh)**2.)
                        Ce = 1./(1.+(np.abs(delta_e)/edge_thresh)**2.)
                        Cw = 1./(1.+(np.abs(delta_w)/edge_thresh)**2.)
                        del delta_n , delta_s , delta_e , delta_w

                if 0.01 > 0:
                        delta_n = (np.append(org_dem_Array[0 , :].reshape(1 , Nj) , org_dem_Array[0 : Ni - 1  , :].reshape(Ni - 1 , Nj) , axis = 0) - org_dem_Array) / resolution
                        delta_s = (np.append(org_dem_Array[1 : Ni  , :].reshape(Ni - 1 , Nj) , org_dem_Array[Ni - 1 , :].reshape(1 , Nj) , axis = 0) - org_dem_Array) / resolution
                        delta_e = (np.append(org_dem_Array[: , 1 : Nj].reshape(Ni , Nj - 1) , org_dem_Array[: , Nj - 1].reshape(Ni , 1) , axis = 1) - org_dem_Array) / resolution
                        delta_w = (np.append(org_dem_Array[: , 0].reshape(Ni , 1) , org_dem_Array[: , 0 : Nj - 1].reshape(Ni , Nj - 1) , axis = 1) - org_dem_Array) / resolution
                        delta_n = np.nan_to_num(delta_n)
                        delta_s = np.nan_to_num(delta_s)
                        delta_e = np.nan_to_num(delta_e)
                        delta_w = np.nan_to_num(delta_w)
                        

                dem_Array = dem_Array +  0.05 * (np.multiply(Cn, delta_n) + np.multiply(Cs, delta_s) + np.multiply(Ce, delta_e) + np.multiply(Cw, delta_w))
                del delta_n , delta_s , delta_e , delta_w , Cn , Cs , Ce , Cw
        
        return dem_Array

def curvature(DEM_file , plan_curvature_file , profile_curvature_file , resolution , option):

    # Option:
    # POLY_CON: ploygon fitting contour curvature
    # POLY_LAP: ploygon fitting laplacian
    # DIFF_CON: finite difference contour curvature
    # POLY_CON: finite difference contour curvature
    
    dem_Array = arcpy.RasterToNumPyArray(DEM_file , nodata_to_value = 0)
    corner = arcpy.Point(arcpy.Describe(DEM_file).Extent.XMin,arcpy.Describe(DEM_file).Extent.YMin)
    dx = arcpy.Describe(DEM_file).meanCellWidth
    Ni = dem_Array.shape[0]
    Nj = dem_Array.shape[1]
    if option == 'POLY_CON'  or  option == 'POLY_LAP':
        L = resolution
        
        Z_2 = np.append(dem_Array[0 , :].reshape(1 , Nj) , dem_Array[0 : Ni - 1  , :].reshape(Ni - 1 , Nj) , axis = 0)
        Z_4 = np.append(dem_Array[: , 0].reshape(Ni , 1) , dem_Array[: , 0 : Nj - 1].reshape(Ni , Nj - 1) , axis = 1)
        Z_6 = np.append(dem_Array[: , 1 : Nj].reshape(Ni , Nj - 1) , dem_Array[: , Nj - 1].reshape(Ni , 1) , axis = 1)
        Z_8 = np.append(dem_Array[1 : Ni  , :].reshape(Ni - 1 , Nj) , dem_Array[Ni - 1 , :].reshape(1 , Nj) , axis = 0)
        
        Z_1 = np.append(Z_2[: , 0].reshape(Ni , 1) , Z_2[: , 0 : Nj - 1].reshape(Ni , Nj - 1) , axis = 1)
        Z_3 = np.append(Z_2[: , 1 : Nj].reshape(Ni , Nj - 1) , Z_2[: , Nj - 1].reshape(Ni , 1) , axis = 1)
        Z_5 = np.copy(dem_Array)
        Z_7 = np.append(Z_8[: , 0].reshape(Ni , 1) , Z_8[: , 0 : Nj - 1].reshape(Ni , Nj - 1) , axis = 1)
        Z_9 = np.append(Z_8[: , 1 : Nj].reshape(Ni , Nj - 1) , Z_8[: , Nj - 1].reshape(Ni , 1) , axis = 1)


        #A = ((Z_1  + Z_3  + Z_7  + Z_9 ) / 4  - (Z_2  + Z_4  + Z_6  + Z_8 ) / 2 + Z_5 ) / L ** 4
        #B = ((Z_1  + Z_3  - Z_7  - Z_9 ) /4 - (Z_2  - Z_8 ) /2) / L ** 3
        #C = ((-Z_1  + Z_3  - Z_7  + Z_9 ) /4 + (Z_4  - Z_6 ) / 2) / L ** 3
        D = ((Z_4  + Z_6 )/2 - Z_5 ) / L ** 2
        E = ((Z_2  + Z_8 ) /2 - Z_5 ) / L ** 2
        F = (-Z_1  + Z_3  + Z_7  - Z_9 ) / (4 * L ** 2)
        G = (-Z_4  + Z_6 ) / (2 * L)
        H = (Z_2  - Z_8 ) / (2 * L)
        #I = Z_5

        del Z_1 , Z_2 , Z_3 ,Z_4 , Z_5 , Z_6 , Z_7 , Z_8 , Z_9

        hx = G
        hxx = 2 * D
        hxy = F
        hy = H
        hyy = 2 * E

        del G , D , F ,H , E
        
        if option == 'POLY_CON':
            curve_Array_plan = (hxx * hy ** 2 - 2 * hxy * hx * hy + hyy * hx ** 2) / ((hx ** 2 + hy ** 2) * (1 + hx ** 2 + hy ** 2) ** 0.5)
            #curve_Array_profile = (hxx * hx ** 2 + 2 * hxy * hx * hy + hyy * hy ** 2) / ((hx ** 2 + hy ** 2) * (1 + hx ** 2 + hy ** 2) ** 1.5)
            #curve_Array_profile = (hxx * hy ** 2 - 2 * hxy * hx * hy + hyy * hx ** 2) / ((hx ** 2 + hy ** 2) ** 1.5)
            curve_Array_profile =  hxx + hyy
        elif option == 'POLY_LAP':
            curve_Array_plan = hxx + hyy

        del hx , hxx , hxy ,hy , hyy
    elif option == 'DIFF_CON'  or  option == 'DIFF_LAP':
        grad_x , grad_y =  np.gradient(dem_Array , resolution)
        grad_xx , _ = np.gradient(grad_x , resolution)
        _, grad_yy = np.gradient(grad_y , resolution)
        if option == 'DIFF_LAP':
            curve_Array = grad_xx + grad_yy
        elif option == 'DIFF_CON':
            grad_xy , _ = np.gradient(grad_y , resolution)
            p =  (grad_x ** 2 + grad_y ** 2)
            q =  1 + p
            curve_Array =  (grad_xx * grad_y ** 2 - 2 * grad_xy * grad_x * grad_y + grad_yy * grad_x ** 2)/ (p * q ** 0.5)
        # Cleaning the borders
    extent = np.where(dem_Array > 0 , 1 , 0).astype(np.int8)
    extent = scipy.ndimage.binary_erosion(extent , iterations = 3).astype(extent.dtype)
    curve_Array_plan = np.where(extent == 1, curve_Array_plan , 0)
    curve_Array_profile = np.where(extent == 1, curve_Array_profile , 0)
    arcpy.NumPyArrayToRaster(curve_Array_plan , corner , dx , dx , value_to_nodata=0 ).save(plan_curvature_file)
    arcpy.NumPyArrayToRaster(curve_Array_profile , corner , dx , dx , value_to_nodata=0 ).save(profile_curvature_file)
            
        

def stream_to_line(in_file_path, out_file_path, stream_raster , flow_dir_raster , out_polyline):
    stream_raster_1 = SetNull(Raster('%s'%in_file_path +  '/' + '%s'%stream_raster) == 0 , 1)
    StreamToFeature (stream_raster_1, '%s'%in_file_path +  '/' + '%s'%flow_dir_raster, '%s'%in_file_path +  '/' + '%s'%out_polyline)

def stream_smooth(in_file_path, out_file_path, in_polyline , out_polyline , tolorance):
    in_polyline = '%s'%in_file_path +  '/' + '%s'%in_polyline
    out_polyline = '%s'%out_file_path +  '/' + '%s'%out_polyline
    arcpy.cartography.SmoothLine(in_polyline, out_polyline, "PAEK", tolorance)

def raster_copy(in_file_path, out_file_path , scale):
    IN_RASTER = Raster('%s'%in_file_path) * scale
    IN_RASTER.save('%s'%out_file_path)


    
def raster_pixel_type(in_file_path, out_file_path, in_raster , out_raster):
    OUT_RASTER = '%s'%out_file_path +  '/' + '%s'%out_raster
    IN_RASTER = Raster('%s'%in_file_path +  '/' + '%s'%in_raster)
    arcpy.CopyRaster_management (IN_RASTER , OUT_RASTER, "", "", "", "", "", "32_BIT_FLOAT" , "", "")
      


def raster_resample(in_file_path, out_file_path , cell_size , in_raster , out_raster):
    
    IN_RASTER = '%s'%in_file_path +  '/' + '%s'%in_raster
    OUT_RASTER = '%s'%out_file_path +  '/' + '%s'%out_raster
    
    arcpy.Resample_management (Raster(IN_RASTER), OUT_RASTER , cell_size , "CUBIC")
           
            
def fill_dem(In_raster, Out_raster , option):
    if option == 'GIS':
        Fill(In_raster).save(Out_raster)
    elif option == 'TAU':
        arcpy.gp.PitRemove (In_raster, 1, Out_raster)
               

def cut_dem(in_file_path, out_file_path , in_raster , bond , out_raster):
        
    IN_RASTER = '%s'%in_file_path +  '/' + '%s'%in_raster
    BOND =  '%s'%in_file_path +  '/' + '%s'%bond
    
    Con(Raster(BOND)  == 1 , Raster(IN_RASTER)).save('%s'%out_file_path +  '/' + '%s'%out_raster)
            


def flow_dir(In_raster, Dir_raster, Slope_raster,  option):
    if option == 'D8':
        FlowDirection(In_raster, "FORCE").save(Dir_raster)
    elif option == 'DINF':
        arcpy.gp.DinfFlowDir (In_raster, 1 , Dir_raster , Slope_raster) 
                 

def flow_acc(Dir_raster , Acc_raster , option):
    if option == 'D8':
        dataType = "INTEGER"
        FlowAccumulation(Dir_raster, "" ,dataType).save(Acc_raster)
    elif option == 'DINF':
        arcpy.gp.AreaDinf (Dir_raster, '' , '' , 0, 1 , Acc_raster)

def flow_acc_weigth(Dir_raster , Wei_raster, Acc_raster , option):
    if option == 'D8':
        dataType = "FLOAT"
        FlowAccumulation(Dir_raster, Wei_raster ,dataType).save(Acc_raster)
            
def flow_acc_bond(in_file_path, out_file_path , acc_tresh , acc_raster , out_raster):
    
    ACC_RASTER = '%s'%in_file_path +  '/' + '%s'%acc_raster
    Con((Raster(ACC_RASTER)  >= acc_tresh) , 1 , 0).save('%s'%out_file_path + '/' + '%s'%out_raster)

            
def flow_acc_max(in_file_path, out_file_path , in_raster, sub , step , deb_mode):
    
    max_acc = arcpy.GetRasterProperties_management('%s'%in_file_path +  '/' + '%s'%in_raster , "MAXIMUM")
    return int(max_acc.getOutput(0))
    


def stream_order(in_file_path, out_file_path , st_raster , dir_raster , out_raster):
    
    ST_RASTER = '%s'%in_file_path +  '/' + '%s'%st_raster
    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%dir_raster
    
    StreamOrder(ST_RASTER , DIR_RASTER).save('%s'%out_file_path +   '/' + '%s'%out_raster)


def CC_dis(in_file_path , curve_raster , ave_k , figure_name):
    CURVE_RASTER = '%s'%in_file_path +  '/' + '%s'%curve_raster
    
    raster = Abs(Raster(CURVE_RASTER))
    curve_Array = arcpy.RasterToNumPyArray(raster)
    corner = arcpy.Point(arcpy.Describe(raster).Extent.XMin,arcpy.Describe(raster).Extent.YMin)
    dx = arcpy.Describe(raster).meanCellWidth

    max_curve = arcpy.GetRasterProperties_management(raster, "MAXIMUM")
    max_curve = float(max_curve.getOutput(0))

    min_curve = arcpy.GetRasterProperties_management(raster, "MINIMUM")
    min_curve = float(min_curve.getOutput(0))

    num_bins = int((max_curve - min_curve) / 0.0005) + 1

    #print max_curve , min_curve , num_bins
    
    cumfreqs, lowlim, binsize, extrapoints = scipy.stats.cumfreq(curve_Array, numbins = num_bins , defaultreallimits=(min_curve , max_curve))

    

    max_cell = np.max(cumfreqs)

    curve_prob = np.zeros((1 , 3))
    for i in range (0 , num_bins):
        k = np.abs(lowlim + binsize * i)
        if k <= 3 * ave_k:
            P = 1 - cumfreqs[i] / max_cell
            curve_prob = np.copy(np.append(curve_prob , np.array([k , P , 0]).reshape(1 , 3) , axis = 0))
    curve_prob = np.delete(curve_prob, 0, 0)
            
    num_point = curve_prob.shape[0]
    for i in range (0 , num_point - 1):
        curve_prob[i , 2] = curve_prob[i , 1] - curve_prob[i + 1 , 1]
    curve_prob = np.delete(curve_prob, num_point - 1, 0)

    threshold = general_linear_fit_log(curve_prob[: , 0] , curve_prob[: , 2] , in_file_path ,figure_name)

    return threshold

def linear_fit(x , y):
    x_bar = np.average(x)
    y_bar = np.average(y)
    xy_bar = np.average(x * y)
    x2_bar = np.average(x ** 2)
    y2_bar = np.average(y ** 2)

    a = (xy_bar  - x_bar * y_bar) / (x2_bar - x_bar ** 2)
    b = y_bar - a * x_bar
    R2 = (xy_bar  - x_bar * y_bar) / ((x2_bar - x_bar ** 2) * (y2_bar - y_bar ** 2)) ** 0.5

    return a , b , R2

def general_linear_fit_log(x , y , out_file_path , figure_name):
    num_point = x.shape[0]
    min_E = 10 ** 10
    for t_i in range (1 , num_point - 1):
        x_1 = x[0 : t_i - 1]
        y_1 = y[0 : t_i - 1]

        x_2 = x[t_i : num_point - 1]
        y_2 = y[t_i : num_point - 1]
        
        a_1 , b_1 , R2_1 = linear_fit(x_1, y_1)
        y_1_s = a_1 * x_1 + b_1
        E_1 = np.sum(np.abs(y_1 - y_1_s))

        a_2 , b_2 , R2_2 = linear_fit(np.log10(x_2), np.log10(y_2))
        y_2_s = 10 ** (a_2 * np.log10(x_2) + b_2)
        E_2 = np.sum(np.abs(y_2 - y_2_s))
        
        if  E_1 + E_2 <= min_E:
            best_t_i = t_i
            best_a_1 = a_1
            best_a_2 = a_2
            best_b_1 = b_1
            best_b_2 = b_2
            min_E = E_1 + E_2

    t_i = best_t_i
    k_thresh = (x[t_i] + x[t_i - 1]) / 2
    fig = pl.figure()
    x_1 = x[0 : t_i - 1]
    y_1 = y[0 : t_i - 1]
    pl.plot(x_1 , y_1 , 'ro')
    pl.plot(x_1 , best_a_1 * x_1 + best_b_1 , 'r-')

    x_2 = x[t_i : num_point - 1]
    y_2 = y[t_i : num_point - 1]
    pl.plot(x_2 , y_2 , 'go')
    pl.plot(x_2 , 10 ** (best_a_2 * np.log10(x_2) + best_b_2) , 'g-')
    pl.plot(np.ones((10 , 1)) * k_thresh , np.arange(0 , 0.1 , 0.01) , 'k--' , linewidth = 2)
    pl.grid(b=True, which='minor', color='k', linestyle='--')
    pl.grid(b=True, which='major', color='k', linestyle='-')
    pl.yticks(fontsize = 16)
    pl.xticks(fontsize = 16)
    pl.xlabel('Curvature (1/m)', fontsize=20)
    pl.ylabel('Probability', fontsize=20)
    fig.savefig('%s'%out_file_path  +  '/' + '%s'%figure_name)
    
    axes = pl.gca()

    return k_thresh
    

def general_linear_fit_lin(x , y , figure_file):
    N = 100
    num_point = x.shape[0]
    min_E = 10 ** 10
    min_x = np.min(x)
    max_x = np.max(x)
    dx = (max_x - min_x) / N 
    for t_i in range (0 , N-1):
        x_thresh = min_x + dx * (t_i + 1)
        x_1 = x[x < x_thresh]
        y_1 = y[x < x_thresh]

        x_2 = x[x >= x_thresh]
        y_2 = y[x >= x_thresh]
        
        a_1 , b_1 , R2_1 = linear_fit(x_1, y_1)
        y_1_s = a_1 * x_1 + b_1
        E_1 = np.mean(np.abs(y_1 - y_1_s))

        a_2 , b_2 , R2_2 = linear_fit(x_2, y_2)
        y_2_s = a_2 * x_2 + b_2
        E_2 = np.mean(np.abs(y_2 - y_2_s))
        
        if  max(E_1 , E_2) / min(E_2 , E_1) <= min_E:
            best_t_i = t_i
            best_a_1 = a_1
            best_a_2 = a_2
            best_b_1 = b_1
            best_b_2 = b_2
            min_E = max(E_1 , E_2) / min(E_2 , E_1)

    t_i = best_t_i
    x_thresh = min_x + dx * (t_i + 1)
    fig = pl.figure()
    x_1 = x[x < x_thresh]
    y_1 = y[x < x_thresh]
    pl.plot(x_1 , y_1 , 'ro')
    pl.plot(x_1 , best_a_1 * x_1 + best_b_1 , 'r-')

    x_2 = x[x >= x_thresh]
    y_2 = y[x >= x_thresh]
    pl.plot(x_2 , y_2 , 'go')
    pl.plot(x_2 , best_a_2 * x_2 + best_b_2 , 'g-')
    #pl.plot(np.ones((10 , 1)) * x_thresh , np.arange(0 , 1 , 0.1) , 'k--' , linewidth = 2)
    pl.grid(b=True, which='minor', color='k', linestyle='--')
    pl.grid(b=True, which='major', color='k', linestyle='-')
    pl.yticks(fontsize = 16)
    pl.xticks(fontsize = 16)
    pl.xlabel('k * Log(A) ', fontsize=20)
    pl.ylabel('Probability', fontsize=20)
    fig.savefig(figure_file)
    
    axes = pl.gca()

    return x_thresh

################################################################################
################################################################################
    
def move_cal(m , i , j):
    if m == 1:
        new_i = i
        new_j = j + 1
    elif m == 2:
        new_i = i + 1
        new_j = j + 1
    elif m == 4:
        new_i = i + 1
        new_j = j 
    elif m == 8:
        new_i = i + 1
        new_j = j - 1
    elif m == 16:
        new_i = i
        new_j = j - 1
    elif m == 32:
        new_i = i - 1
        new_j = j - 1
    elif m == 64:
        new_i = i - 1
        new_j = j
    elif m == 128:
        new_i = i - 1
        new_j = j + 1
    else:
        new_i = i
        new_j = j
    return new_i, new_j

#############################################################################
#############################################################################

def work_check(sub):
    ORG_DEM = 'C:/miladhooshyar/Spring2014/GIS/archydro_curve/Valley_Network/CDEM' + '/'+ 'subDEM_' +  str(sub) + '.tif'
    TT = True
    while TT == True:
        try:
            org_ext = arcpy.Describe(ORG_DEM).Extent.XMin
            env_path = env.workspace
            ext = env.extent
            sk_path = env.scratchWorkspace
            TT = False
        except:
            print 'Fail_Workspace'
            TT = True
    return os.path.basename(env_path) == str(sub) , org_ext == ext.XMin , os.path.basename(sk_path) == 'Temp' + str(sub)

#############################################################################
#############################################################################

def mod_area_curve(area_curve):
    num_point = area_curve.shape[0]
    mod_area_curve = np.zeros((1 , 2))
    
    cur_size = area_curve[0 , 0]
    i = 0
    ave_curve = 0
    counter = 0
    while i < num_point - 1:
        #print area_curve[i , 0] , cur_size
        if area_curve[i , 0] == cur_size:
            ave_curve = ave_curve + area_curve[i , 1]
            counter = counter + 1
            i = i + 1
        else:
            ave_curve = ave_curve / counter
            mod_area_curve = np.copy(np.append(mod_area_curve , np.array([cur_size , ave_curve]).reshape(1 , 2) , axis = 0))
            #print cur_size , ave_curve
            ave_curve = 0
            counter = 0
            cur_size = area_curve[i , 0]
            
        
        
    mod_area_curve = np.delete(mod_area_curve, 0 , 0 )

    return mod_area_curve
            
#############################################################################
#############################################################################



###################################################################################################
###################################################################################################

###################################################################################################
###################################################################################################

def Dinf_to_D8(Dinf_acc_raster , D8_raster , Acc_raster):
    Ele_raster = 1 / Log10(Raster(Dinf_acc_raster) + 1)
    fill_dem(Ele_raster, os.path.dirname(Dinf_acc_raster) + '\\Temp_fill.tif' ,  'GIS')
    flow_dir(os.path.dirname(Dinf_acc_raster) + '\\Temp_fill.tif', D8_raster, '',  'D8')
    flow_acc(D8_raster , Acc_raster , 'D8')
    Ele_raster = None
    os.unlink(os.path.dirname(Dinf_acc_raster) + '\\Temp_fill.tif')
    
    
###################################################################################################
###################################################################################################
def Dinf_fix(Dinf_raster , Curve_raster , Fix_Dinf_raster):
    def downstream_cells(d):
        d = round(d , 4)
        pi_4 = round( np.pi / 4 , 4)
        if d > 0 and d < pi_4:
            a1 = 1
            a2 = 128
        elif d > pi_4 and d < 2 * pi_4:
            a1 = 128
            a2 = 64
        elif d > 2 * pi_4 and d < 3 * pi_4:
            a1 = 64
            a2 = 32
        elif d > 3 * pi_4 and d < 4 * pi_4:
            a1 = 32
            a2 = 16
        elif d > 4 * pi_4 and d < 5 * pi_4:
            a1 = 16
            a2 = 8
        elif d > 5 * pi_4 and d < 6 * pi_4:
            a1 = 8
            a2 = 4
        elif d > 6 * pi_4 and d < 7 * pi_4:
            a1 = 4
            a2 = 2
        elif d > 7 * pi_4 and d < 8 * pi_4:
            a1 = 2
            a2 = 1
        elif d == 0:
            a1 = 1
            a2 = 1
        elif d == pi_4:
            a1 = 128
            a2 = 128
        elif d == 2 * pi_4:
            a1 = 64
            a2 = 64
        elif d == 3 * pi_4:
            a1 = 32
            a2 = 32
        elif d == 4 * pi_4:
            a1 = 16
            a2 = 16
        elif d == 5 * pi_4:
            a1 = 8
            a2 = 8
        elif d == 6 * pi_4:
            a1 = 4
            a2 = 4
        elif d == 7 * pi_4:
            a1 = 2
            a2 = 2
        else:
            a1 = -10
            a2 = -10
        return a1 , a2



    def direction_curvature(org_dir , curve1 , curve2):
        if curve1 > 0 and curve2 > 0 and curve1 != curve2:
            if curve1 > curve2:
                alfa_m = org_dir - (np.pi / 4) * int(org_dir / (np.pi / 4))
                new_alfa_m = alfa_m * (curve2 / curve1) 
                new_dir = (np.pi / 4) * int(org_dir / (np.pi / 4)) + new_alfa_m
            else:
                alfa_m = (np.pi / 4) * int(org_dir / (np.pi / 4) + 1) - org_dir
                new_alfa_m = alfa_m * (curve1 / curve2) 
                new_dir = (np.pi / 4) * int(org_dir / (np.pi / 4) + 1) - new_alfa_m
        elif curve1 > 0 and curve2 < 0:
            new_dir = (np.pi / 4) * int(org_dir / (np.pi / 4))
        elif curve1 < 0 and curve2 > 0:
            new_dir = (np.pi / 4) * int(org_dir / (np.pi / 4) + 1)
        elif curve1 != -10 or curve2 != 10:
            new_dir = org_dir
        else:
            new_dir = org_dir
        return new_dir

    def move_cal(m , i , j , min_i , max_i , min_j , max_j):
        if m == 1:
            new_i = i
            new_j = j + 1
        elif m == 2:
            new_i = i + 1
            new_j = j + 1
        elif m == 4:
            new_i = i + 1
            new_j = j 
        elif m == 8:
            new_i = i + 1
            new_j = j - 1
        elif m == 16:
            new_i = i
            new_j = j - 1
        elif m == 32:
            new_i = i - 1
            new_j = j - 1
        elif m == 64:
            new_i = i - 1
            new_j = j
        elif m == 128:
            new_i = i - 1
            new_j = j + 1
        else:
            new_i = i
            new_j = j
        if new_i < min_i or new_i >= max_i:
            new_i = i
            
        if new_j < min_j or new_j >= max_j:
            new_j = j
        return new_i, new_j

    curve = arcpy.RasterToNumPyArray(Curve_raster, nodata_to_value = -10).astype(np.float32)
    Dinf = arcpy.RasterToNumPyArray(Dinf_raster, nodata_to_value = -10)
    corner = arcpy.Point(arcpy.Describe(Dinf_raster).Extent.XMin,arcpy.Describe(Dinf_raster).Extent.YMin)
    dx = arcpy.Describe(Dinf_raster).meanCellWidth

    number_row = Dinf.shape[0]
    number_col = Dinf.shape[1]

    downstream_cells = np.vectorize(downstream_cells)
    dir_1 , dir_2 = downstream_cells(Dinf)

    move_cal = np.vectorize(move_cal)
    i_1 , j_1 = move_cal(dir_1 , np.arange(0 , number_row).reshape(number_row , 1) , np.arange(0 , number_col).reshape(1 , number_col) , 0 , number_row , 0 , number_col)
    i_2 , j_2 = move_cal(dir_2 , np.arange(0 , number_row).reshape(number_row , 1) , np.arange(0 , number_col).reshape(1 , number_col), 0 , number_row , 0 , number_col)

    curve1 = curve[i_1 , j_1]
    curve2 = curve[i_2 , j_2]

    direction_curvature = np.vectorize(direction_curvature ,otypes = [np.float32])
    new_Dinf = direction_curvature(Dinf , curve1 , curve2)


    new_Dinf = np.where(new_Dinf == -1 , Dinf , new_Dinf)
    new_Dinf = np.where(Dinf == -10 , -10 , new_Dinf)
    arcpy.NumPyArrayToRaster(new_Dinf , corner, dx , dx , value_to_nodata=-10).save(Fix_Dinf_raster)


def find_head(in_stream_Array , dir_Array):

    number_row = dir_Array.shape[0]
    number_col = dir_Array.shape[1]
    heads_list = [] 

    cx = scipy.sparse.coo_matrix(in_stream_Array)
    
    for ii , jj , v in zip(cx.row, cx.col, cx.data):
        #print ii , jj , v
        flage_no_head = 0
        for i in range (-1 , 2):
            for j in range (-1 , 2):
                new_i = ii + i 
                new_j = jj + j
                if bond_check(new_i , 0 , number_row) == 1 and bond_check(new_j , 0 , number_col):
                    if in_stream_Array[new_i , new_j] == 1:
                        move = dir_Array[new_i , new_j]
                        if bond_check(move , 1 , 129) == 1:
                            next_i, next_j = move_cal(move, new_i , new_j)
                            if (next_i, next_j) == (ii , jj):
                                flage_no_head = 1

        if flage_no_head == 0:
            heads_list.append((ii , jj))
            
    #del in_stream_Array , dir_Array , cx
    
    return heads_list

#####################################################################################
#####################################################################################

def find_end(in_stream_Array , heads_list , dir_Array):
    
    ends_list = []
    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]

    for point in heads_list:
        new_i = point[0]
        new_j = point[1]
        last_point = point
        flage_end = 0
        while flage_end == 0:
            move = dir_Array[new_i , new_j]
            new_i, new_j = move_cal(move, new_i , new_j)
            if bond_check(new_i , 0 , num_row) == 0 or bond_check(new_j , 0 , num_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
                flage_end = 1
            elif in_stream_Array[new_i , new_j] != 1:
                flage_end = 1
                if (any((p == (new_i , new_j)) for p in ends_list)) == False:
                    ends_list.append((new_i , new_j))
            else:
                last_point = (new_i , new_j)
                    
    #del in_stream_Array , dir_Array , heads_list , 
    return ends_list

#############################################################################################################
#############################################################################################################

def find_head_block(in_stream_Array , order_Array , dir_Array):
    number_row = dir_Array.shape[0]
    number_col = dir_Array.shape[1]
    heads_list = [] 

    cx = scipy.sparse.coo_matrix(in_stream_Array)

    for ii , jj , v in zip(cx.row, cx.col, cx.data):
        #print ii , jj , v
        flage_no_head = 0
        order = order_Array[ii , jj]
        for i in range (-1 , 2):
            for j in range (-1 , 2):
                new_i = ii + i 
                new_j = jj + j
                if bond_check(new_i , 0 , number_row) == 1 and bond_check(new_j , 0 , number_col):
                    if in_stream_Array[new_i , new_j] == 1 and order_Array[new_i , new_j] == order:
                        move = dir_Array[new_i , new_j]
                        if bond_check(move , 1 , 129) == 1:
                            next_i, next_j = move_cal(move, new_i , new_j)
                            if (next_i, next_j) == (ii , jj):
                                flage_no_head = 1

        if flage_no_head == 0:
            heads_list.append((ii , jj))
            
    #del in_stream_Array , dir_Array , cx
    
    return heads_list
    
#####################################################################################
#####################################################################################

def connect_stream_smart(in_file_path , out_file_path, stream_raster , flow_dir_raster , curve_raster , out_raster , connect_ratio):
    total_num_add = 0

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster
    CURVE_RASTER = '%s'%in_file_path +  '/' + '%s'%curve_raster

 
    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER)
    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    curve_Array = arcpy.RasterToNumPyArray(CURVE_RASTER , nodata_to_value=0)
    corner = arcpy.Point(arcpy.Describe(DIR_RASTER).Extent.XMin,arcpy.Describe(DIR_RASTER).Extent.YMin)
    dx = arcpy.Describe(DIR_RASTER).meanCellWidth
                   
    
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    
    

    heads_list = find_head(Sp_stream_Array , dir_Array)

##    head_Array = np.zeros_like(stream_Array)   
##    p = 1
##    for index in heads_list:
##        head_Array[index] = p
##        p = p + 1
##    heads_raster = arcpy.NumPyArrayToRaster(head_Array,corner, dx ,dx)
##    heads_raster.save('%s'%out_file_path +  '/heads.tif')

    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]

    list_end = []
    num_end = np.zeros_like(curve_Array)
    visited = np.zeros_like(stream_Array)

    p = -1
    for index in heads_list:
        p = p + 1
        #print p
        list_passed , last_point , length , length_cuve , visited = move_forward_connect_1(index , stream_Array , dir_Array , visited , curve_Array , num_row , num_col)
        #list_passed , last_point , length = move_forward_connect_2(index , stream_Array , dir_Array , num_row , num_col)
        if (any((p == last_point) for p in list_end)) == False:
            list_end.append(last_point)
        num_end[last_point] = num_end[last_point] + length_cuve

##    curve_end = np.zeros_like(curve_Array)   
##    for index in list_end:
##        curve_end[index] = num_end[index]
##    heads_raster = arcpy.NumPyArrayToRaster(curve_end,corner, dx ,dx)
##    heads_raster.save('%s'%out_file_path +  '/end_curve.tif')
        
    #arcpy.NumPyArrayToRaster(num_end ,corner,dx , dx , value_to_nodata=0).save('%s'%out_file_path +  '/len.tif')
    num_add = 1
    while num_add > 0:
        num_add = 0
        for index in list_end:
##            label_st = label_Array[index]
            list_passed_gap , last_point_gap , length_gap , length_curve_gap = move_forward_gap(index , stream_Array , dir_Array ,curve_Array , num_row , num_col)
##            max_gap = (num_end[index] + length_curve_gap) * connect_ratio
            max_gap = (num_end[index]) * connect_ratio
            

##            if num_end[index] <= 100:
##                label_end = label_Array[last_point_gap]
##            else:
##                label_end = -1
                
            if length_gap > 0 and length_gap <= max_gap:# and length > 2:
                for point in list_passed_gap:
                    if stream_Array[point] == 0:
                        stream_Array[point] = 1
                        num_add = num_add + 1
                        total_num_add = total_num_add + 1
                _ , last_point , _ = move_forward_connect_2(index , stream_Array , dir_Array , num_row , num_col)
                num_end[last_point] = num_end[last_point] - float(length_gap) / connect_ratio + num_end[index] #+ length_curve_gap
                list_end.remove(index)
                
    arcpy.NumPyArrayToRaster(stream_Array,corner,dx , dx , value_to_nodata=0).save('%s'%out_file_path +  '/' + '%s'%out_raster)
            

    
    
    return total_num_add


def move_forward_gap(index , Sp_stream_Array , dir_Array ,curve_array , number_row , number_col): 
    length = 1
    length_curve = 0
    list_passed = []
    new_i = index[0]
    new_j = index[1]
    last_point = index
    flage_end = 0
    while flage_end == 0:
        list_passed.append((new_i , new_j))
        move = dir_Array[new_i][new_j]
        new_i, new_j = move_cal(move, new_i , new_j)

        if bond_check(new_i , 0 , number_row) == 0 or bond_check(new_j , 0 , number_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
            #last_point = (new_i , new_j)
            flage_end = 1
        elif Sp_stream_Array[new_i , new_j] != 0:
            #last_point = (new_i , new_j)
            flage_end = 1
        else:
            last_point = (new_i , new_j)
            length = length + 1
            length_curve = length_curve + curve_array[new_i , new_j]
    return list_passed , last_point , length , length_curve

def move_forward_connect_1(index , Sp_stream_Array , dir_Array , visited , curve_array, number_row , number_col): 
    length_curve = 0
    length = 0
    list_passed = []
    new_i = index[0]
    new_j = index[1]
    last_point = index
    flage_end = 0
    while flage_end == 0:
        list_passed.append((new_i , new_j))
        move = dir_Array[new_i][new_j]
        new_i, new_j = move_cal(move, new_i , new_j)

        if bond_check(new_i , 0 , number_row) == 0 or bond_check(new_j , 0 , number_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
            #last_point = (new_i , new_j)
            flage_end = 1
        elif Sp_stream_Array[new_i , new_j] != 1:
            #last_point = (new_i , new_j)
            flage_end = 1
        else:
            last_point = (new_i , new_j)
            if visited[new_i , new_j] == 0:
                length_curve = length_curve + curve_array[new_i , new_j]
                length = length + 1
            visited[new_i , new_j] = 1
    return list_passed , last_point , length , length_curve , visited

def move_forward_connect_2(index , Sp_stream_Array , dir_Array , number_row , number_col): 
    length = 1
    list_passed = []
    new_i = index[0]
    new_j = index[1]
    last_point = index
    flage_end = 0
    while flage_end == 0:
        list_passed.append((new_i , new_j))
        move = dir_Array[new_i][new_j]
        new_i, new_j = move_cal(move, new_i , new_j)

        if bond_check(new_i , 0 , number_row) == 0 or bond_check(new_j , 0 , number_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
            #last_point = (new_i , new_j)
            flage_end = 1
        elif Sp_stream_Array[new_i , new_j] != 1:
            #last_point = (new_i , new_j)
            flage_end = 1
        else:
            last_point = (new_i , new_j)
            length = length + 1    
    return list_passed , last_point , length

##########################################################################################################
##########################################################################################################
def bond_check(i , min_i , max_i):
    flage = 0
    if i >= min_i and i < max_i:
        flage = 1
    return flage
    
###############################################################################################################
###############################################################################################################     

def delet_isolated_stream(in_file_path , out_file_path, stream_raster , order_raster , flow_dir_raster  , acc_raster , min_length_iso , min_length_con , out_raster):
    
    num_deleted = 0

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ACC_RASTER = '%s'%in_file_path +  '/' + '%s'%acc_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster


    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value = 0)
    acc_Array = arcpy.RasterToNumPyArray(ACC_RASTER , nodata_to_value = 0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value = 0)
    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    corner = arcpy.Point(arcpy.Describe(DIR_RASTER).Extent.XMin,arcpy.Describe(DIR_RASTER).Extent.YMin)
    dx = arcpy.Describe(DIR_RASTER).meanCellWidth
            
    Sp_stream_Array = sparse.csr_matrix(stream_Array)

    number_row = stream_Array.shape[0]
    number_col = stream_Array.shape[1]

    heads_list = find_head(Sp_stream_Array , dir_Array)

    
    for index in heads_list:
        list_passed , last_point , flage_connected_end , ave_curve , length = move_forward(index , Sp_stream_Array , dir_Array , dir_Array , order_Array , number_row , number_col)
        if flage_connected_end == 0 and length <= min_length_iso:
            for point in list_passed:
                stream_Array[point] = 0
                num_deleted = num_deleted + 1
        if flage_connected_end == 1 and length <= min_length_con:
            for point in list_passed:
                stream_Array[point] = 0
                num_deleted = num_deleted + 1

    arcpy.NumPyArrayToRaster(stream_Array,corner,dx , dx , value_to_nodata=0).save('%s'%out_file_path +  '/' + '%s'%out_raster)
       
    return num_deleted

########################################################################################
########################################################################################

def move_backward(dir_Array , Sp_stream_Array , ele_Array  , order_Array, list_passed , last_point , number_row , number_col , ele_thresh , length_thresh , head):
    alt_stream_list = [[]]
    #num_alt_st = 1
    alt_st = 0
    list_passed_new = []
    flage_done = 0
    list_passed_temp = []
    length_list = []
    while flage_done == 0:
        if alt_st > 0:
            alt_stream_list. append([])
        flage_done = 1
        flage_out = 0
        position = last_point

        length = 0

        temp_ele_thresh = ele_thresh
        temp_length_thresh = length_thresh

##        if head == 299:
##            print '000'
        while flage_out == 0:
            conut_found = 0
            for i in range (-1 , 2):
                for j in range (-1 , 2):
                    new_i = position[0] + i 
                    new_j = position[1] + j
                    if  bond_check(new_i , 0 , number_row) == 1 and bond_check(new_j , 0 , number_col) == 1:
                        move = dir_Array[new_i , new_j]
                        if bond_check(move , 1 , 129) == 1:
                            next_i, next_j = move_cal(move, new_i , new_j)
                            if Sp_stream_Array[new_i , new_j] == 1 and (next_i, next_j) == position and \
                               (any((p == (new_i , new_j)) for p in list_passed)) == False and (any((p == (new_i , new_j)) for p in list_passed_new)) == False:
                                conut_found = conut_found + 1
                                next_position = (new_i , new_j)
                                if order_Array[next_position] == 1 and length == 1:
                                    temp_ele_thresh = 10 ** 10
                                    temp_length_thresh = 10 ** 10
                                elif order_Array[next_position] != 1:
                                    temp_ele_thresh = ele_thresh
                                    temp_length_thresh = length_thresh
                                
                            
            if conut_found > 1:
                #num_alt_st = num_alt_st + 1
                flage_done = 0
                list_passed_temp = []
                list_passed_temp.append(next_position)
                
##            if head == 138:
##                    print conut_found , temp_ele_thresh , ele_Array[next_position]

            if conut_found > 0 and ele_Array[position] <= temp_ele_thresh and length <= temp_length_thresh :          
                position = next_position
                alt_stream_list[alt_st].append(position)
                list_passed_temp.append(position)
                length = length + 1
            else:
                flage_out = 1
                length_list.append(length)
                
                alt_st = alt_st + 1
                for p in list_passed_temp:
                    list_passed_new.append(p)
    return alt_stream_list , len(alt_stream_list) , length_list

########################################################################################
########################################################################################
            
def move_forward(index , Sp_stream_Array , dir_Array , curve_Array , order_Array , number_row , number_col): 
    length = 1
    list_passed = []
    new_i = index[0]
    new_j = index[1]
    last_point = index
    flage_end = 0
    ave_curve = 0
    flage_connected_end = 0
    while flage_end == 0:
        list_passed.append((new_i , new_j))
        ave_curve = ave_curve + curve_Array[new_i , new_j]
        move = dir_Array[new_i][new_j]
        new_i, new_j = move_cal(move, new_i , new_j)

        if bond_check(new_i , 0 , number_row) == 0 or bond_check(new_j , 0 , number_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
            #last_point = (new_i , new_j)
            flage_end = 1
        elif Sp_stream_Array[new_i , new_j] != 1:
            #last_point = (new_i , new_j)
            flage_end = 1
        elif order_Array[new_i , new_j] > 1:
            last_point = (new_i , new_j)
            flage_end = 1
            flage_connected_end = 1
        else:
            last_point = (new_i , new_j)
            length = length + 1
            
    return list_passed , last_point , flage_connected_end , (ave_curve / length) , length

        
########################################################################################
########################################################################################
def stream_delet_small_fast(in_file_path , out_file_path , ridge_raster , stream_raster , dem_raster , flow_dir_raster , acc_raster , order_raster , curve_raster , out_raster):

    num_deleted = 0

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ACC_RASTER = '%s'%in_file_path +  '/' + '%s'%acc_raster
    DEM_RASTER = '%s'%in_file_path +  '/' + '%s'%dem_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    CURVE_RASTER = '%s'%in_file_path +  '/' + '%s'%curve_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster
    RIDGE_RASTER = '%s'%in_file_path +  '/' + '%s'%ridge_raster


    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value=0)
    acc_Array = arcpy.RasterToNumPyArray(ACC_RASTER , nodata_to_value=0)
    ele_Array = arcpy.RasterToNumPyArray(DEM_RASTER , nodata_to_value=0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value=0)
    curve_Array = arcpy.RasterToNumPyArray(CURVE_RASTER , nodata_to_value=0)
    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    ridge =  arcpy.RasterToNumPyArray(RIDGE_RASTER , nodata_to_value=0)

    corner = arcpy.Point(arcpy.Describe(DIR_RASTER).Extent.XMin,arcpy.Describe(DIR_RASTER).Extent.YMin)
    dx = arcpy.Describe(DIR_RASTER).meanCellWidth
        
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    
    
    number_row = dir_Array.shape[0]
    number_col = dir_Array.shape[1]

    heads_list = find_head(Sp_stream_Array , dir_Array)

##    head_Array = np.zeros_like(stream_Array)   
##    p = 0
##    for index in heads_list:
##        p = p + 1
##        head_Array[index] = p
##    heads_raster = arcpy.NumPyArrayToRaster(head_Array,corner, dx ,dx)
##    heads_raster.save('%s'%out_file_path +  '/heads.tif')


    p = 0
    for index in heads_list:
        p = p + 1
        if stream_Array[index] == 1:
            flage_delet = 0
            list_passed , last_point , flage_connected_end , _ , length = move_forward(index , Sp_stream_Array , dir_Array , curve_Array , order_Array , number_row , number_col)
            list_passed.reverse()
            if flage_connected_end == 1:
                alt_stream_list , num_alt_st , alt_length_list = move_backward(dir_Array , Sp_stream_Array , ele_Array , order_Array , list_passed , last_point , number_row , number_col , (ele_Array[index] + 20) , (1.25 * length)  , p)

                
##                if p == 397:
##                    temp_st = np.zeros_like(curve_Array)
##                    for alt_st in range (0 , num_alt_st):
##                        for point in alt_stream_list[alt_st]:
##                            temp_st[point] = curve_Array[point]
##                    temp_st_raster = arcpy.NumPyArrayToRaster(temp_st,corner, dx ,dx)
##                    temp_st_raster.save('%s'%out_file_path +  '/temp_st.tif')
##
##                    temp_st = np.zeros_like(curve_Array)
##                    for point in list_passed:
##                        temp_st[point] = curve_Array[point]
##                    temp_st_raster = arcpy.NumPyArrayToRaster(temp_st,corner, dx ,dx)
##                    temp_st_raster.save('%s'%out_file_path +  '/temp_st_o.tif')
                
                #print 'Head = ' , p  , num_alt_st

                
                count_alt = 0
##                if p == 115:
##                    temp_st = np.zeros_like(dir_Array)
                for alt_st in range (0 , num_alt_st):
                    count_alt = count_alt + 1
                    curve_alt = 0
                    count_sep = 0
                    flage_half = 0
##                    list_dual_point = []
##                    conuter_ave_curve = 0
                    if flage_delet == 0:
                        pp = 0
                        win_point = 0
                        sum_curve_1 = 0
                        sum_curve_2 = 0
                        flage_ridge_found = 0
                        list_same_1 = []
                        list_same_2 = []
                        for point in list_passed:
##                            min_diff = 100
##                            for alt_point in alt_stream_list[alt_st]:
##                                ele_diff = abs(ele_Array[point]- ele_Array[alt_point])
##                                #print point , alt_point #ele_diff , min_diff
##                                if ele_diff <= min_diff:
##                                    dual_point = alt_point
##                                    min_diff = ele_diff
##                            if min_diff <= 100:

##                            if p == 78:
##                                for xxx in alt_stream_list[alt_st]:
##                                    temp_st[xxx] = count_alt
                            if pp < len(alt_stream_list[alt_st]):
                                dual_point = alt_stream_list[alt_st][pp]
##                                if p == 115:
##                                    temp_st[dual_point] = pp
##                                    temp_st[point] = pp
                                d_i = dual_point[0] - point[0]
                                d_j = dual_point[1] - point[1]
                                for dt in range (0 , 30):
                                    new_i = int(point[0] + round(float(dt) / 29 * float(d_i)))
                                    new_j = int(point[1] + round(float(dt) / 29 * float(d_j)))
                                    mid_point = (new_i , new_j)
                                    if ridge[mid_point] == 1:
                                        flage_ridge_found = 1
                                if flage_ridge_found == 0:
                                    list_same_1.append(point)
                                    list_same_2.append(dual_point)
                                    if curve_Array[point] > curve_Array[dual_point]:
                                        win_point = win_point + 1
                                    sum_curve_1 = sum_curve_1 + curve_Array[point]
                                    sum_curve_2 = sum_curve_2 + curve_Array[dual_point] 
                                pp = pp + 1     
                        flage_delete_stream = 0
                        flage_delete_alt_stream = 0

                        ## Deleting cells
                        if len(list_same_1) > 0 and len(list_same_2) > 0:
                            order_st = order_Array[list_same_1[-1]]
                            order_alt_st = order_Array[list_same_2[-1]]
##                            if p == 1146 or p == 1142 or p == 1137:
##                                print p , len(list_same_1) , len(list_same_2) , win_point , length, alt_length_list[alt_st] , order_st , order_alt_st
                            # higher order wins
                            if order_alt_st > order_st:
                                flage_delete_stream = 1
                            # they are both first order
                            else:
                                # if one is extramly smaller than the other
                                if length <= 0.5 * alt_length_list[alt_st]:
                                    flage_delete_stream = 1
                                elif alt_length_list[alt_st] <= 0.5 * length:
                                    flage_delete_alt_stream = 1
                                #they are in same size
                                elif win_point < 0.5 * min(len(list_same_1) , len(list_same_2)):
                                    flage_delete_stream = 1
                                elif win_point == 0.5 * min(len(list_same_1) , len(list_same_2)) and sum_curve_2 > sum_curve_1:
                                    flage_delete_stream = 1
                                else:
                                    flage_delete_alt_stream = 1
##                            if p == 1146 or p == 1142 or p == 1137:
##                                print p , flage_delete_stream , flage_delete_alt_stream        
                        if flage_delete_stream == 1:
                            flage_delet = 1
    ##                        for point in list_same_1:
    ##                            stream_Array[point] = 0
    ##                            num_deleted = num_deleted + 1
                            if flage_ridge_found == 1:
                                for point in list_same_1:
                                    stream_Array[point] = 0
                                    num_deleted = num_deleted + 1
                            else:
                                for point in list_passed:
                                    stream_Array[point] = 0
                                    num_deleted = num_deleted + 1
                        if flage_delete_alt_stream == 1:
    ##                        for point in list_same_2:
    ##                            if order_Array[point] == 1:
    ##                                stream_Array[point] = 0
    ##                                num_deleted = num_deleted + 1
                            if flage_ridge_found == 1:
                                for point in list_same_2:
                                    if order_Array[point] == 1:
                                        stream_Array[point] = 0
                                        num_deleted = num_deleted + 1
                            else:
                                for point in alt_stream_list[alt_st]:
                                    if order_Array[point] == 1:
                                        stream_Array[point] = 0
                                        num_deleted = num_deleted + 1                           
##                if p == 115:
##                    temp_st_raster = arcpy.NumPyArrayToRaster(temp_st,corner, dx ,dx)
##                    temp_st_raster.save('%s'%out_file_path +  '/temp_st_o_1.tif')                          
    arcpy.NumPyArrayToRaster(stream_Array,corner,dx , dx , value_to_nodata=0).save('%s'%out_file_path +  '/' + '%s'%out_raster)
    
    return num_deleted
########################################################################################
########################################################################################

def stream_extention(in_file_path , out_file_path  , stream_raster , old_stream_raster , dem_raster , flow_dir_raster , acc_raster , order_raster , old_order_raster , curve_raster , out_raster):

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ACC_RASTER = '%s'%in_file_path +  '/' + '%s'%acc_raster
    DEM_RASTER = '%s'%in_file_path +  '/' + '%s'%dem_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    OLD_ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%old_order_raster
    CURVE_RASTER = '%s'%in_file_path +  '/' + '%s'%curve_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster
    OLD_STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%old_stream_raster


    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value=0)
    acc_Array = arcpy.RasterToNumPyArray(ACC_RASTER , nodata_to_value=0)
    ele_Array = arcpy.RasterToNumPyArray(DEM_RASTER , nodata_to_value=0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value=0)
    old_order_Array = arcpy.RasterToNumPyArray(OLD_ORDER_RASTER , nodata_to_value=0)
    curve_Array = arcpy.RasterToNumPyArray(CURVE_RASTER , nodata_to_value=0)
    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    old_stream_Array = arcpy.RasterToNumPyArray(OLD_STREAM_RASTER , nodata_to_value=0)

    corner = arcpy.Point(arcpy.Describe(DIR_RASTER).Extent.XMin,arcpy.Describe(DIR_RASTER).Extent.YMin)
    dx = arcpy.Describe(DIR_RASTER).meanCellWidth
        
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    old_Sp_stream_Array = sparse.csr_matrix(old_stream_Array)
    
    
    number_row = dir_Array.shape[0]
    number_col = dir_Array.shape[1]

    heads_list = find_head(Sp_stream_Array , dir_Array)

    for index in heads_list:
        list_passed , last_point , flage_connected_end , ave_curve , length = move_forward(index , Sp_stream_Array , dir_Array , curve_Array , order_Array , number_row , number_col)
        if flage_connected_end == 1:
            alt_stream_list , num_alt_st , alt_length_list = move_backward(dir_Array , old_Sp_stream_Array , ele_Array , old_order_Array , list_passed , index , number_row , number_col , (ele_Array[index] + 1) , (1.25 * length)  , 0)
            if num_alt_st > 0:
                max_curve = 0
                for alt_st in range (0 , num_alt_st):
                    if alt_length_list[alt_st] > 1:
                        count = 0
                        ave_curve = 0
                        for point in alt_stream_list[alt_st]:
                            ave_curve = ave_curve + curve_Array[point]
                            count = count + 1
                        ave_curve = ave_curve / count

                        if ave_curve > max_curve:
                            best_alt_st = alt_st
                            max_curve =  ave_curve
                if  max_curve > 0:       
                    for point in alt_stream_list[best_alt_st]:
                        stream_Array[point] = 1
                    
    arcpy.NumPyArrayToRaster(stream_Array , corner , dx , dx , value_to_nodata=0).save('%s'%out_file_path +  '/' + '%s'%out_raster)
           
####################################################################################################################
####################################################################################################################

def find_por_point(in_file_path , out_file_path , stream_raster , dem_raster ,  flow_dir_raster , acc_raster  , order_raster , out_text):

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    DEM_RASTER = '%s'%in_file_path +  '/' + '%s'%dem_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster
    ACC_RASTER = '%s'%in_file_path +  '/' + '%s'%acc_raster
    

    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value=0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value=0)
    dem_Array = arcpy.RasterToNumPyArray(DEM_RASTER , nodata_to_value=0)
    acc_Array = arcpy.RasterToNumPyArray(ACC_RASTER , nodata_to_value=0)
    
    dsc=arcpy.Describe(DIR_RASTER)
    ext=dsc.Extent
    corner=arcpy.Point(ext.XMin,ext.YMin)
    
    X_min = dsc.EXTENT.XMin
    Y_min = dsc.EXTENT.YMin

    X_max = dsc.EXTENT.XMax
    Y_max = dsc.EXTENT.YMax

    dy = dsc.meanCellHeight
    dx = dsc.meanCellWidth
        
        
    #Step 1: Find channel heads and elevation
    heads_list = find_head(Sp_stream_Array , dir_Array)
    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]
    
    channel_head = []
    channel_ele = []
    for index in heads_list:
        channel_head.append(index)
        channel_ele.append(dem_Array[index])
    
    
    #Step 2: find por points
    por_point = []
    por_ele = []
    p = -1
    for index in heads_list:
        p = p + 1
        new_i = index[0]
        new_j = index[1]
        flage_end = 0
        list_passed = []
        length = 0
        while flage_end == 0:
            list_passed.append((new_i , new_j))
            move = dir_Array[new_i , new_j]
            new_i, new_j = move_cal(move, new_i , new_j)
            
            if bond_check(new_i , 0 , num_row) == 0 or bond_check(new_j , 0 , num_col) == 0 or bond_check(move , 1 , 129) == 0 or bond_check(dir_Array[new_i , new_j] , 1 , 129) == 0:
                flage_end = 1
            elif order_Array[new_i , new_j] != 1: 
                last_point = (new_i , new_j)
                flage_end = 1
            else:
                length = length + 1
        
        por_point.append(list_passed[length - 2])
        por_ele.append(dem_Array[list_passed[length - 2]])

    file_por = open('%s' %out_file_path + '/por_point.txt','w')
    number_point = len(por_point)
    for p in range (0 , number_point):
        file_por.write('%f '%(X_min + por_point[p][1] * dx + dx / 2))
        file_por.write('% f '%(Y_max - por_point[p][0] * dy - dy / 2))
        file_por.write('% f '%(channel_ele[p] + 2))
        file_por.write('% f '%(por_ele[p]))
        file_por.write('% f '%(X_min + channel_head[p][1] * dx + dx / 2))
        file_por.write('% f '%(Y_max - channel_head[p][0] * dy  - dy / 2))
        file_por.write('% d \n'%p)
    file_por.close()

########################################################################
########################################################################

def unchannel_delet(in_file_path , out_file_path , stream_raster ,  flow_dir_raster ,  order_raster , Chan_Unchan , Unch_thresh , out_raster):
    
    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster

    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value=0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value=0)
    
    dsc=arcpy.Describe(DIR_RASTER)
    ext=dsc.Extent
    corner=arcpy.Point(ext.XMin,ext.YMin)
    

    #Step 1: Find channel heads and elevation
    heads_list = find_head(Sp_stream_Array , dir_Array)
    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]
    
    
    #Step 2: find por points

    p = - 1

    num_delet = 0

    for index in heads_list:
        p = p + 1
        if Chan_Unchan[p , 2] < Unch_thresh:
            num_delet = num_delet + 1
            new_i = index[0]
            new_j = index[1]
            flage_end = 0
            list_passed = []
            length = 0
            while flage_end == 0:
                list_passed.append((new_i , new_j))
                                    
                move = dir_Array[new_i , new_j]
                new_i, new_j = move_cal(move, new_i , new_j)

                if bond_check(new_i , 0 , num_row) == 0 or bond_check(new_j , 0 , num_col) == 0 or bond_check(move , 1 , 129) == 0:
                    flage_end = 1
                elif order_Array[new_i , new_j] != 1:
                    last_point = (new_i , new_j)
                    flage_end = 1
                else:
                    length = length + 1
            for point in list_passed:
                stream_Array[point] = 0
    

    cleaned_st_raster = arcpy.NumPyArrayToRaster(stream_Array , corner,dsc.meanCellWidth,dsc.meanCellHeight , value_to_nodata=0)
    cleaned_st_raster.save('%s'%out_file_path +  '/' + '%s'%out_raster)
    #cleaned_st_raster = None
            
    return num_delet

########################################################################################################################################################
########################################################################################################################################################

def head_delete(in_file_path , out_file_path , dem_raster , stream_raster ,  flow_dir_raster ,  order_raster , head_ele , out_raster):
    #env.workspace = in_file_path

    DIR_RASTER = '%s'%in_file_path +  '/' + '%s'%flow_dir_raster
    ORDER_RASTER = '%s'%in_file_path +  '/' + '%s'%order_raster
    DEM_RASTER = '%s'%in_file_path +  '/' + '%s'%dem_raster
    STREAM_RASTER = '%s'%in_file_path +  '/' + '%s'%stream_raster

    stream_Array = arcpy.RasterToNumPyArray(STREAM_RASTER , nodata_to_value=0)
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    dir_Array = arcpy.RasterToNumPyArray(DIR_RASTER , nodata_to_value=0)
    order_Array = arcpy.RasterToNumPyArray(ORDER_RASTER , nodata_to_value=0)
    dem_Array = arcpy.RasterToNumPyArray(DEM_RASTER , nodata_to_value=0)
    
    dsc=arcpy.Describe(DIR_RASTER)
    ext=dsc.Extent
    corner=arcpy.Point(ext.XMin,ext.YMin)

    #Step 1: Find channel heads and elevation
    heads_list = find_head(Sp_stream_Array , dir_Array)
    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]
    
    
    #Step 2: find por points

    p = - 1

    for index in heads_list:
        p = p + 1
        new_i = index[0]
        new_j = index[1]
        flage_end = 0
        list_passed = []
        length = 0
        list_passed.append(index)
        if head_ele[p , 0] > 0:
            while flage_end == 0:
                move = dir_Array[new_i , new_j]
                new_i, new_j = move_cal(move, new_i , new_j)

                if bond_check(new_i , 0 , num_row) == 0 or bond_check(new_j , 0 , num_col) == 0 or bond_check(move , 1 , 129) == 0:
                    flage_end = 1
                elif order_Array[new_i , new_j] != 1:
                    last_point = (new_i , new_j)
                    flage_end = 1
                else:
                    length = length + 1

                    if dem_Array[new_i , new_j] > head_ele[p , 0]:
                        list_passed.append((new_i , new_j))
                        
            for point in list_passed:
                stream_Array[point] = 0
    

    cleaned_st_raster = arcpy.NumPyArrayToRaster(stream_Array , corner,dsc.meanCellWidth,dsc.meanCellHeight , value_to_nodata=0)
    cleaned_st_raster.save('%s'%out_file_path +  '/' + '%s'%out_raster)

########################################################################################################################################################
########################################################################################################################################################

def find_por_point_simple(in_file_path , out_file_path , stream_raster ,  dir_raster , acc_raster , scale):
    env.workspace = in_file_path
    
    dsc=arcpy.Describe(stream_raster)
    sp_ref=dsc.SpatialReference
    ext=dsc.Extent
    corner=arcpy.Point(ext.XMin,ext.YMin)
    
    X_min = dsc.EXTENT.XMin
    Y_min = dsc.EXTENT.YMin

    X_max = dsc.EXTENT.XMax
    Y_max = dsc.EXTENT.YMax

    dy = dsc.meanCellHeight
    dx = dsc.meanCellWidth
    
    stream_Array = arcpy.RasterToNumPyArray(stream_raster , nodata_to_value=0)
    Sp_stream_Array = sparse.csr_matrix(stream_Array)
    
    dir_Array = arcpy.RasterToNumPyArray(dir_raster)
    acc_Array = arcpy.RasterToNumPyArray(acc_raster)

        

    #Step 1: Find channel heads and elevation
    heads_list = find_head(Sp_stream_Array , dir_Array)
    num_row = dir_Array.shape[0]
    num_col = dir_Array.shape[1]

    number_point = 0
    channel_head = []
    for point in heads_list:
        print 'SubBasin' , number_point , ',' , 'Area = ' , int(float(acc_Array[point]) * scale)
        channel_head.append((X_min + point[1] * dx + dx / 2 , Y_max - point[0] * dy  - dy / 2))
        number_point += 1


    return channel_head , number_point
