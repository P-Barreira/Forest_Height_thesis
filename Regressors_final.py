import rasterio
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
import random
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import path_names as pn

# ---------------------------------------------------------------

def read_bands(name_location,GRD_SLC):

    files = pn.file_path(name_location,GRD_SLC) # GLCM; image
    
    if GRD_SLC == True: # GRD
         
        n_bands = 40 
        band_list = []
        for numb,dataset in enumerate(files): 
            if numb == 1:
                n_bands = 6
                original_img = rasterio.open(dataset)
            band_list+=[rasterio.open(dataset).read(i) for i in range(1, n_bands + 1)]
            
    else: # SLC
        
        n_bands = 10
        band_list = []
        for numb,dataset in enumerate(files):
            if numb == 2:
                n_bands = 2
                original_img = rasterio.open(dataset)
                
            band_list+=[rasterio.open(dataset).read(i) for i in range(1, n_bands + 1)]
            
    return band_list,original_img

# ---------------------------------------------------------------
def PCA_calc(band_df,GRD_SLC):
    
    band_array = np.array(band_df)
    
    num_bands, n_rows = band_array.shape
    reshaped_data = band_array.transpose(1,0)
    
    if GRD_SLC == True:
        reshaped_data = reshaped_data[-6:-2] #only gamma0 and sigma0 
        aux_name = '_GRD_'
    else:   
        reshaped_data = reshaped_data[-4:] #only coherence and phase
        aux_name = '_SLC_'
    
    reshaped_data = reshaped_data.transpose(1,0)

    n_components = 4 
    pca = PCA(n_components=n_components)

    pca_result = pca.fit_transform(reshaped_data)

    pca_df = pd.DataFrame(
        data=pca_result, 
        columns=['PC'+aux_name+str(i) for i in range(1,pca_result.shape[1]+1)])

    return pca_df

# ---------------------------------------------------------------
def create_dataframe_location(band_list,GRD_SLC):
    band_array = np.array(band_list)
    num_bands, height,width = band_array.shape
    
    reshaped_data = band_array.transpose(1, 2, 0).reshape(-1, num_bands)

    col_names = pn.band_names(GRD_SLC)
    
    df= pd.DataFrame(
        data=reshaped_data, 
        columns=col_names)
    
    return df, (height,width)

# ---------------------------------------------------------------
def create_dataframe(path,name):
    band = [rasterio.open(path).read(1)]
    band_a = np.array(band)
    reshaped_band_a = band_a.transpose(1, 2, 0).reshape(-1, 1)
    
    df_aux = pd.DataFrame(
        data=reshaped_band_a, 
        columns=[name])
    
    del band,band_a,reshaped_band_a

    return df_aux

# ---------------------------------------------------------------
def calc_temp_comp(location,GRD_SLC,n_images):
    
    print('Temporal Composite')
    
    path_aux = pn.band_location_name(location,GRD_SLC)
    
    aux, original_img = read_bands(path_aux,GRD_SLC)
    aux = np.array(aux)
        
    _, height,width = np.array(aux).shape
    
    range_aux = pn.range_bands(n_images,GRD_SLC)
        
    for i in range_aux:
        path_aux = pn.band_location_name(location,GRD_SLC,i)
        
        band_list, _ = read_bands(path_aux,GRD_SLC)
        band_list = np.array(band_list)
        aux += band_list
        
    aux = np.array(aux)
    mean_values = aux/n_images
        
    df,_ = create_dataframe_location(mean_values,GRD_SLC)
        
    del mean_values, aux, band_list
    
    return df, (height,width), original_img

# ---------------------------------------------------------------
def row_list(orig_array,WL,orig_row_width,U_D,L_R):
    # 7x7 example: U_D ranges from -3 to 3 and L_R ranges from 0 to 3; if 0 then same line (U_D) or same column (L_R)
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
def spacial_filter_with_LCLU(aux_list, aux_LCLU, info,window):
    
    list_a = np.array(aux_list.copy())
        
    # change the kernel based on the LCLU info
    aux_1 = np.isin(aux_LCLU,info) # checks if a value of LCLU is in 'info' and returns an array with the same shape with True/False in each element
    
    aux_1 = aux_1.transpose(1,0)
    aux_img = list_a.transpose(1,0)
    
    window_limit = math.floor(window/2)
    
    val = []
    
    for idx, i in enumerate(aux_img):
        
        LCLU_kernel = aux_1[idx].reshape((window,window))
        img_kernel= i.reshape((window,window))
        
        LCLU_kernel_center = LCLU_kernel[window_limit,window_limit]
        
        if (LCLU_kernel_center == False) & (LCLU_kernel.any()):
            smallest_dist_index = (-1,-1)
            smallest_dist = math.dist(smallest_dist_index,(window_limit,window_limit))
            
            for idx1 in range(0,window):
                for idx2 in range(0,window):
                    
                    if LCLU_kernel[idx1,idx2] == True:
                        distance = math.dist((idx1,idx2),(window_limit,window_limit))
                        
                        if distance < smallest_dist:
                            smallest_dist = distance
                            smallest_dist_index = (idx1,idx2)
                            if smallest_dist == 1:
                                break
            
            if smallest_dist_index == (-1,-1): # if none value in kernel is in info, no change is made
                val.append(img_kernel[window_limit,window_limit]) 
            else:
                val.append(img_kernel[smallest_dist_index])      
        else:
            val.append(img_kernel[window_limit,window_limit])
    
    del aux_1,LCLU_kernel,img_kernel,aux_img, list_a
        
    return val

# ---------------------------------------------------------------
def create_array_for_spacial_filter(img_array,Width,window,LCLU_array,LCLU_info):
    
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
                    
        aux = spacial_filter_with_LCLU(aux_list,aux_LCLU,LCLU_info,window)
        
        result.append(aux)
          
    del aux, aux_list, aux_LCLU    
    return np.array(result)

# ---------------------------------------------------------------
def spatial_filter(df_all_bands, original_shape,window,path_LCLU,info,GRD_SLC):
    
    LCLU_image = rasterio.open(path_LCLU).read(1)
    
    LCLU_reshaped = LCLU_image.reshape(-1)
    
    band_array = np.array(df_all_bands)
    height = original_shape[0]
    width = original_shape[1]
    
    reshaped_data = band_array.transpose(1,0)
    num_bands, n_rows = reshaped_data.shape
    
    a_list = []
    
    for i in range(0,num_bands):
        aa = reshaped_data[i]
        #comment line below if no LCLU is to be applied
        #aa = create_array_for_spacial_filter(aa,width,window,LCLU_reshaped,info)
        aa = aa.reshape((height,width))
        aa = ndimage.median_filter(aa,size=window)
        a_list.append(aa) 
    
    band_array = np.array(a_list)
    
    reshaped_data = band_array.transpose(1, 2, 0).reshape(-1, num_bands)

    col_names = pn.band_names(GRD_SLC)
    
    df_filtered = pd.DataFrame(
        data=reshaped_data, 
        columns=col_names)

    return df_filtered

# ---------------------------------------------------------------
def Regressor_calc(df_bands,pca_df,df_LCLU,df_CHM,pred_df1,pred_df2,location,LCLU_info):
    df = df_bands
    df = df.join(pca_df)
    df = df.join(df_LCLU)
    df = df.join(df_CHM)
        
    df_set = df.dropna()
    df_set = df_set[df_set['LCLU'].isin(LCLU_info)]
    
    df_set = df_set[(df_set['CHM'] > 0)]
    
    dados = df_set.drop(columns=['CHM'])
    
    print(len(df_set))
    
    classe = df_set['CHM']
    r2=[]
    RMSE=[]
    RMSE_p=[]
    MAE=[]
    r2_2=[]
    RMSE_2=[]
    RMSE_p_2=[]
    MAE_2=[]
    
    pred_df1['index'] = pd.Series(np.nan,index=classe.index)
    pred_df1.set_index('index')
    
    pred_df2['index'] = pd.Series(np.nan,index=classe.index)
    pred_df2.set_index('index')
    
    for i in range(0,10):
        rng = random.randint(0,1000)
        print(i)
        X_train,X_test,y_train,y_test=train_test_split(dados,classe,test_size=0.75, random_state=rng)  
        
        RF_reg = RandomForestRegressor(n_estimators=400, max_depth=30, n_jobs=-1, random_state=rng) 
        XGB_reg = XGBRegressor(device='cuda', booster= 'gbtree', n_estimators=1900 , max_depth=9, eta=0.1, subsample=1, random_state=rng)
        
        estimators=[('RF', RF_reg),('XGB',XGB_reg)]
        
        reg = StackingRegressor( estimators=estimators,final_estimator=LinearRegression())
        
        reg = XGBRegressor(device='cuda', booster= 'gbtree', n_estimators=100 , max_depth=3, eta=0.1, subsample=1, random_state=rng)
                
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        
        y_pred = list(map(lambda x: 0 if x<0 else x, y_pred)) 
        
        r2 += [r2_score(y_test, y_pred)]
    
        MSE = mean_squared_error(y_test, y_pred)
 
        RMSE += [math.sqrt(MSE)]
    
        RMSE_p += [math.sqrt(MSE)/np.mean(y_test)]
    
        MAE += [mean_absolute_error(y_test, y_pred)]
        
        pred_df1[location+'_'+str(i+1)] = pd.Series(y_pred,index=y_test.index)
        
        del MSE, y_pred
        
        print('//')
                
        y_pred = reg.predict(X_train)
        
        y_pred = list(map(lambda x: 0 if x<0 else x, y_pred))
                
        r2_2 += [r2_score(y_train, y_pred)]
    
        MSE = mean_squared_error(y_train, y_pred)
 
        RMSE_2 += [math.sqrt(MSE)]
    
        RMSE_p_2 += [math.sqrt(MSE)/np.mean(y_train)]
    
        MAE_2 += [mean_absolute_error(y_train, y_pred)]
        
        pred_df2[location+'_'+str(i+1)] = pd.Series(y_pred,index=y_train.index)
        
        y_train_len = len(y_train)
        y_test_len = len(y_test)
        
        del MSE, y_pred
        
        del X_train,X_test,y_train,y_test
    
    
    del dados, classe
    del df
    del df_set
    
    return r2, RMSE, RMSE_p, MAE, r2_2, RMSE_2, RMSE_p_2, MAE_2, y_train_len, y_test_len

# ---------------------------------------------------------------
def Predict(forest_type, country, window):
    '''
    forest_type: \n 
            1 -> Forest\n
            2 -> Shrub\n
            3 -> Forest+Shrub\n\n
    
    country: \n 
            1 -> Portugal\n
            2 -> Spain\n
            3 -> California\n\n
    
    window: size of spatial kernel\n
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
    
    excel_path_name = pn.path_name_excel(forest_type) 
    
    with pd.ExcelWriter(excel_path_name[0]) as writer_1_75p:
        with pd.ExcelWriter(excel_path_name[1]) as writer_1_25p:
            location_names = pn.location_names(country)

            for location in location_names:
                print(location)
                
                predictions_csv_path = pn.csv_path_name(location)
                
                CHM_path = location_names[location][0] 
                LCLU_path = location_names[location][1] 
                n_images_GRD = location_names[location][2]
                n_images_SLC = location_names[location][3]
                        
                df_LCLU = create_dataframe(LCLU_path,'LCLU')
                df_CHM = create_dataframe(CHM_path,'CHM')
                        
                df_bands_GRD, original_bands_shape_GRD, original_img_GRD = calc_temp_comp(location,True,n_images_GRD) # GRD -> True; SLC -> False
                df_bands_SLC, original_bands_shape_SLC, original_img_SLC = calc_temp_comp(location,False,n_images_SLC)
                        
                print('Spacial Filter GRD')
                df_bands_GRD = spatial_filter(df_bands_GRD,original_bands_shape_GRD,window,LCLU_path,info_LCLU,True)
                print('Spacial Filter SLC')
                df_bands_SLC = spatial_filter(df_bands_SLC,original_bands_shape_SLC,window,LCLU_path,info_LCLU,False)
                        
                pca_df_GRD = PCA_calc(df_bands_GRD,True)
                pca_df_SLC = PCA_calc(df_bands_SLC,False)
                        
                predictions_75_df = pd.DataFrame()
                predictions_25_df = pd.DataFrame()
                                
                df_bands = df_bands_GRD
                df_bands = df_bands_SLC
                df_bands = df_bands.join(df_bands_SLC, lsuffix='_left', rsuffix='_right')
                        
                pca_df = pca_df_GRD
                pca_df = pca_df.join(pca_df_SLC, lsuffix='_left', rsuffix='_right')
                        
                r2, RMSE, RMSE_p, MAE, r2_2, RMSE_2, RMSE_p_2, MAE_2, train_lenght, test_lenght = Regressor_calc(df_bands,pca_df,df_LCLU,df_CHM,predictions_75_df,predictions_25_df,location,info_LCLU)
                            
                Results_df = pd.DataFrame({'Samples': test_lenght ,'R2': list(map(lambda x: x*100,r2)),'RMSE':RMSE,'RMSE%':list(map(lambda x: x*100,RMSE_p)),'MAE':MAE})
                                
                Means = Results_df.mean()                        
                Means_df = pd.DataFrame([Means],columns=Results_df.columns, index=['Mean'])
                            
                Results_df = pd.concat([Results_df,Means_df])
                                
                predictions_mean = predictions_75_df.mean(axis=1)
                        
                df_CHM['Prediction'] = predictions_mean
                predictions_75_df['Mean'] = predictions_mean
                                                        
                Results_df.to_excel(excel_writer=writer_1_75p,sheet_name=location)
                
                predictions_75_df.to_csv(predictions_csv_path[0],index=True)
                
                del Means
                del Results_df,Means_df
                del predictions_75_df
                
                #25p
                Results_df = pd.DataFrame({'Samples': train_lenght ,'R2': list(map(lambda x: x*100,r2_2)),'RMSE':RMSE_2,'RMSE%':list(map(lambda x: x*100,RMSE_p_2)),'MAE':MAE_2})
                                
                Means = Results_df.mean()
                Means_df = pd.DataFrame([Means],columns=Results_df.columns, index=['Mean'])
                Results_df = pd.concat([Results_df,Means_df])
                                
                predictions_mean = predictions_25_df.mean(axis=1)
                        
                df_CHM['Prediction'] = predictions_mean
                predictions_25_df['Mean'] = predictions_mean
                        
                Results_df.to_excel(excel_writer=writer_1_25p,sheet_name=location)
                        
                predictions_25_df.to_csv(predictions_csv_path[1],index=True)
                        
                del Means
                del Results_df,Means_df
                del predictions_25_df, predictions_mean
    
    return True


if __name__ == "__main__":
    forest_type = 1
    country = 1
    window = 7
    Predict(forest_type, country, window)