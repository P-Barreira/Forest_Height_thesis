import rasterio
import numpy as np
import pandas as pd
import math
import random
from scipy import ndimage
import path_names as pn

# ---------------------------------------------------------------
def save_GTiff(changed_image, original_image, save_name):
    
    if len(changed_image.shape) == 3:
        n_bands = changed_image.shape[0]
        img_height=changed_image.shape[1]
        img_width=changed_image.shape[2]
    else:
        n_bands = 1
        img_height=changed_image.shape[0]
        img_width=changed_image.shape[1]
    
    new_image = rasterio.open(
        save_name+'.tif',
        'w',
        driver="GTiff",
        height=img_height,
        width=img_width,
        count=n_bands,
        dtype=changed_image.dtype,
        crs= original_image.crs,
        transform=original_image.transform,
        nodata=0 
    )
    
    new_image.write(changed_image, list(range(1,n_bands+1)))
    new_image.close()

# ---------------------------------------------------------------
def calc_Lee(aux_list, window):
    
    list_a = np.array(aux_list.copy())
    
    x_center = list_a[math.floor((len(list_a))/2)]
        
    sigma_v = 0.26
    
    mean = np.mean(list_a,axis=0) 
    
    variance_z = (((list_a-mean)**2)/(window**2)).sum(axis=0)
        
    variance_x = (variance_z-(mean**2)*(sigma_v**2))/((sigma_v**2)+1)
        
    b_denom = ((mean**2)*(sigma_v**2) + (1+(sigma_v**2))*variance_x)+1e-20
    
    b = variance_x/b_denom
    
    b = np.where(b<0,0,b) 
    
    val = mean + b * (x_center - mean)
    
    return val

# ---------------------------------------------------------------
def calc_Lee_border(aux_list, aux_LCLU, info):
    
    list_a = np.array(aux_list.copy())
    
    x_center = list_a[math.floor((len(list_a))/2)]
    
    LCLU_center = aux_LCLU[math.floor((len(aux_LCLU))/2)]
    
    sigma_v = 0.26
    
    # change the kernel based on the LCLU info
    
    aux_1 = np.isin(aux_LCLU,info).astype(int) # checks if a value of LCLU is in 'info' and returns an array with the same shape with 1/0 in each element
    
    aux_center = np.isin(LCLU_center,info)
    
    list_a = np.where((aux_1 == 0) & (aux_center == True), -10, list_a) # initially was 0, but since water is also 0, this changed to -10 just to be sure
    
    mean = np.mean(list_a,axis=0,where=list_a != -10)
    
    aux_1 = list_a.transpose(1,0)
    variance_z = [] 
    for idx, _ in enumerate(aux_1):
        
        i = np.delete(aux_1[idx] ,np.argwhere(aux_1[idx] == -10)) # removes the values == -10 from each array
        
        variance_z.append(sum(((i-mean[idx])**2)/(len(i)))) # since each array has a diferent lenght
    
    variance_x = (variance_z-(mean**2)*(sigma_v**2))/((sigma_v**2)+1)
        
    b_denom = ((mean**2)*(sigma_v**2) + (1+(sigma_v**2))*variance_x)+1e-20
    
    b = variance_x/b_denom
    
    b = np.where(b<0,0,b) 
    
    val = mean + b * (x_center - mean)
    
    return val

# ---------------------------------------------------------------
def row_list(orig_array,WL,orig_row_width,U_D,L_R):
    # U_D ranges from -3 to 3 and L_R ranges from 0 to 3; if 0 then same line (U_D) or same column (L_R)
    result = list(orig_array[(WL)+orig_row_width*U_D:(WL)+orig_row_width*(U_D+1)])
    
    # if in center column
    if L_R == 0:
        return result
    
    if L_R < 0:
        L_R = -L_R
        for L in range(1,L_R+1):
            result.insert(0,result[0])
            result.pop(len(result)-1)
    else:
        for L in range(1,L_R+1):
            result.append(result[len(result)-1])
            result.pop(0)
            
    return result

# ---------------------------------------------------------------
def apply_filter(img_array:np.array,Width,window,LCLU_array,LCLU_info,border_check=False):
    
    window_limit = math.floor(window/2) # distance from center to kernel limit
    
    img = img_array.copy()
    LCLU = LCLU_array.copy()
    
    # these steps are for the top and bottom images' borders
    img=list(img)
    img[0:0] = (img[0:Width])*window_limit
    img[len(img):len(img)] = (img[-Width:])*window_limit
    
    LCLU=list(LCLU)
    LCLU[0:0] = (LCLU[0:Width])*window_limit
    LCLU[len(LCLU):len(LCLU)] = (LCLU[-Width:])*window_limit
    
    result=[]
    aux_list = []
    aux_LCLU = []
    begin_flg = True
    for WL in range(Width*window_limit,len(img)-Width*window_limit,Width): 
        
        if begin_flg == True:
            for ud in range(-window_limit,window_limit+1):
                for lr in range(-window_limit,window_limit+1):
                    aux_list.append(row_list(img,WL,Width,U_D=ud,L_R=lr))
                    aux_LCLU.append(row_list(LCLU,WL,Width,U_D=ud,L_R=lr))
            begin_flg = False
        else:
            
            # remove the top row of the kernels
            for i in range(0,window):
                aux_list.pop(0)
                aux_LCLU.pop(0)
            
            ud=window_limit
            # only add the bottom row of the kernels
            for lr in range(-window_limit,window_limit+1):
                aux_list.append(row_list(img,WL,Width,U_D=ud,L_R=lr))
                aux_LCLU.append(row_list(LCLU,WL,Width,U_D=ud,L_R=lr))
                    
        if border_check:
            #print('With borders')
            aux = calc_Lee_border(aux_list,aux_LCLU,LCLU_info)
        else:
            aux = calc_Lee(aux_list,window)
        
        result.append(aux)
          
    del aux, aux_list, aux_LCLU    
    return np.array(result)

# ---------------------------------------------------------------
def speckle_filter(path_img:str,path_LCLU:str,LCLU_info, GTiff_Save = False, window = 3, with_border = False):
    
    result_image = []
    
    original_img_raw = rasterio.open(path_img)
    LCLU_image = rasterio.open(path_LCLU).read(1)
    
    original_img = original_img_raw.read()
    
    band_number = 0
    
    print('Applying',str(window)+'x'+str(window),'filter')
        
    for img_band in original_img:
            
        band_number+=1
        print(band_number)
            
        height,width = img_band.shape
        
        # change arrays from 2D to 1D
        reshaped_band = img_band.reshape(-1)
        reshaped_LCLU = LCLU_image.reshape(-1)
            
        # change pixel information based (or not) on LCLU (only forest/shrub/both)
            
        aux_array = apply_filter(reshaped_band,width,window,reshaped_LCLU,LCLU_info,with_border)
        
        aux_array = aux_array.reshape(height,width)
            
        aux_array = aux_array.astype('float32')
            
        result_image.append(aux_array)
        
    result_image = np.array(result_image)
    
    if GTiff_Save:
        save_name = pn.save_name_Gtiff(path_img)
        save_GTiff(result_image,original_img_raw,save_name)
    
    return result_image

# ---------------------------------------------------------------
def begin_filtering(forest_type, country, window, border_on):
    '''
    forest_type: \n 
            1 -> Forest\n
            2 -> Shrub\n
            3 -> Forest+Shrub\n\n
    
    country: \n 
            1 -> Portugal\n
            2 -> Spain\n
            3 -> California\n\n
    
    window: size of spatial kernel\n\n
    
    border_on:\n 
            True -> change borders
            False -> no change
    '''
    
    if country == 1:
        print('Portugal')
        if forest_type == 1:
            info_LCLU = list(range(5111,5124))
        elif forest_type == 2:
            info_LCLU = list(range(6111,6112))
        else:
            info_LCLU = list(range(5111,5124))+list(range(6111,6112))
            
    elif country == 2:
        print('Spain')
        if forest_type == 1:
            info_LCLU = list(range(23,26))
        elif forest_type == 2:
            info_LCLU = list(range(26,30))
        else:    
            info_LCLU = list(range(23,30))
                        
    else:
        print('California')
        if forest_type == 1:
            info_LCLU = list(range(41,44))
        elif forest_type == 2:
            info_LCLU = list(range(52,53))
        else:    
            info_LCLU = list(range(41,44))+list(range(52,53))
            
    location_names = pn.location_names(country)
    
    for location in location_names:
        lclu_path = location_names[location][1]
        img_number = location_names[location][2]
        
        for idx in range(1,img_number+1):
            path_img = pn.path_image_to_filter(location,idx)
            
            speckle_filter(path_img,lclu_path,LCLU_info=info_LCLU,GTiff_Save=True,window=window,with_border=border_on)
    
    return True

if __name__ == "__main__":
    forest_type = 1
    country = 1
    window = 7
    border_on = True
    begin_filtering(forest_type, country, window, border_on)