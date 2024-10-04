# Plotting scree plot for PLS models with an increasing number of latent variables 
global pls_dict
pls_dict = {}
def PLS_scree(Xc, Yc, Xp, Yp, k, label = 'Objective variable'):
    ''' Function to draw to scree plot and the explained variance when fitting PLS model to x and y
        Xc: array: (number_of_samples,number_of_variables)
            Absorbance of samples in Calibration dataset. 
        Yc: array: (number_of_samples,1)
            Objective variable in Validation dataset.
        Xp: array: (number_of_samples,number_of_variables)
            Absorbance of samples in Calibration dataset. 
        Yp: array: (number_of_samples,1)
            Objective variable in Validation dataset.
        k: 
            Number of folds in cross-validation 
        label: optional 
            name of Y-block 
        
        IMPORTANT: the RMSE and R2 in every iteration of LVs number are stored in the pls_dict dictionary variable. 
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import r2_score
    
    np.random.seed(1)
    totalLV=np.arange(1,21,1) # Default: maximum 20 LV
    fig=plt.figure(figsize=(48,18))
    fig.suptitle('Root Mean Squared of Error & Coefficient of Determination when fitting '+label,size=45)
    
    RMSE_cal_PLS=np.squeeze(np.zeros((1,20)))
    RMSE_cv_PLS=np.squeeze(np.zeros((1,20)))
    RMSE_p_PLS=np.squeeze(np.zeros((1,20)))
    R2_cal_PLS=np.squeeze(np.zeros((1,20)))
    R2_cv_PLS=np.squeeze(np.zeros((1,20)))
    R2_p_PLS=np.squeeze(np.zeros((1,20)))
    
    for numLV in totalLV:
        pls = PLSRegression(n_components=numLV)
        pls.fit(Xc, Yc)
        # Calibration 
        scores=cross_validate(pls,Xc,Yc,scoring=['explained_variance','neg_root_mean_squared_error','r2'],cv=k)  # Default: K-fold cross-validation, for other CV methods: visit main site of scikit-learn 
        R2_cal_PLS[numLV-1]=r2_score(Yc,pls.predict(Xc))

        # Cross-validation
        RMSE_cv_PLS[numLV-1]=-np.mean(scores['test_neg_root_mean_squared_error'])
        R2_cv_PLS[numLV-1]=np.mean(scores['test_r2'])

        # External Validation
        RMSE_p_PLS[numLV-1]=mean_squared_error(Yp, pls.predict(Xp))
        R2_p_PLS[numLV-1]=r2_score(Yp,pls.predict(Xp))
    
    
    totalLV=np.arange(1,21,1)
    ax0=fig.add_subplot(121)
    ax1=fig.add_subplot(122)
    ax0.plot(totalLV,np.squeeze(RMSE_cv_PLS),marker='.',markersize=10,color='b',label='Cross-validation')
    ax0.plot(totalLV,np.squeeze(RMSE_cal_PLS),marker='.',markersize=10,color='r',label='Calibration')
    ax0.plot(totalLV,np.squeeze(RMSE_p_PLS),marker='.',markersize=10,color='k',label='Prediction')
    ax1.plot(totalLV,np.squeeze(R2_cv_PLS),marker='.',markersize=10,color='b',label='Cross-validation')
    ax1.plot(totalLV,np.squeeze(R2_cal_PLS),marker='.',markersize=10,color='r',label='Calibration') 
    ax1.plot(totalLV,np.squeeze(R2_p_PLS),marker='.',markersize=10,color='k',label='Prediction') 
    ax0.set_xticks(totalLV)
    ax1.set_xticks(totalLV)
    ax0.tick_params(axis='both', which='major', labelsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax0.set_title('RSME',size=40,pad=15)
    ax1.set_title('R2',size=40,pad=15)
    ax1.legend(prop={'size': 35})
    ax0.legend(prop={'size': 35})
    ax0.set_xlabel('Latent variables',size=45,labelpad=15)
    ax1.set_xlabel('Latent variables',size=45,labelpad=15)
    
    ax0.plot(np.array((RMSE_cv_PLS.tolist().index(RMSE_cv_PLS.min())+1,RMSE_cv_PLS.tolist().index(RMSE_cv_PLS.min())+1)), 
             np.array((0,RMSE_cv_PLS.min())),
               linestyle = '--',lw=2,c='r')
    
    pls_dict[label+' RMSE_cv']=RMSE_cv_PLS
    pls_dict[label+' RMSE_cal']=RMSE_cal_PLS
    pls_dict[label+' RMSE_p']=RMSE_p_PLS
    pls_dict[label+' R2_cv']=R2_cv_PLS
    pls_dict[label+' R2_cal']=R2_cal_PLS
    pls_dict[label+' R2_p']=R2_p_PLS
