import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

# Want this to be rejected, p < 0.05
def adf_test(timeseries, verbose=False):
    #print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    
    return dftest[1]
    
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if verbose:
        print(dfoutput)
    

# Want this to be null, p > 0.05
def kpss_test(timeseries, verbose=False):
    #print("Results of KPSS Test:")
    kpsstest = kpss(timeseries, regression="c", nlags="auto")
    
    
    return kpsstest[1]
    
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    if verbose:
        print(kpss_output)

def lag(col, k=1):
    return col.shift(k)
    
def add_ps(df, dct, trt_lag, add_L, non_add_L_set, include_time=True, logistic=False, include_month=True, verbose=False):

    #prop_score_str = '{X} ~ lag({X},1) + lag({Y}, 1)'.format(X=dct['X'][0],Y=dct['Y'][0]) 

    #if logistic:
    #    prop_score_str = '{X} ~ '.format(X=dct['X'][0])      
    
    #prop_score_str = '{X} ~ {X}.shift(1) + {Y}.shift(1)'.format(X=dct['X'][0],Y=dct['Y'][0]) 

    #if include_time:
    #    prop_score_str += '+ Time'
    #if include_month:
    #    prop_score_str += ' + C(Month)'

    #print('non_add_L_set: ', non_add_L_set)
        
    # Add confounds
    """
    if 'C' in dct:
        for conf in dct['C']:
            #if logistic:
            #    prop_score_str += ' + lag({C}, 0)'.format(C=conf[0]) # + lag({C}, 0)                
                
            #else:
            if add_L and (conf[0] not in non_add_L_set):
                #print(conf, add_L and (conf not in non_add_L_set))
                prop_score_str += ' + lag({C}, 1) + lag({C}, 0)'.format(C=conf[0]) # + lag({C}, 0)
            else:
                prop_score_str += ' + lag({C}, 1)'.format(C=conf[0])
            #prop_score_str += ' + {C}.shift(1) + {C}.shift(0)'.format(C=conf[0]) # + lag({C}, 0)
    """

    # Binary case only
    assert 'Ind_PS' in dct and 'Dep_PS' in dct
    
    prop_score_str = '{Dep} ~ '.format(Dep=dct['Dep_PS'][0])
    if include_time:
        prop_score_str += '+ Time'
    if include_month:
        prop_score_str += ' + C(Month)'
    for var in dct['Ind_PS']:
        prop_score_str += ' + lag({C}, {lag})'.format(C=var[0], lag=var[1][0]) # + lag({C}, 0)                

          
    if verbose:
        print('ps string: ', prop_score_str)

    #print('df.shape: ', df.shape)
    #df.to_csv('prop_df.csv')
        
    if logistic:
        #print('df columns: ', df.columns)
        prop_mod = smf.logit(formula = prop_score_str, data = df)        
    else:
        prop_mod = smf.ols(formula = prop_score_str, data = df)
    
    
    
    prop_res = prop_mod.fit()

    if verbose:
        print('prop score summary: ')
        print(prop_res.summary())

    df['PS'] = prop_res.predict(df)


    if 'C' in dct:
        dct['C'].append(('PS', [trt_lag]))
    else:
        dct['C'] = [('PS', [trt_lag])]
        
    return df, dct

def run_ols(df, dct, fit_method, trt_lag, include_time=True, include_month=True, verbose=False):
    analysis_str = '{Y} ~ '.format(Y=dct['Y'][0])

    for key in dct:
        if key in ['X','Y']:
            for this_lag in dct[key][1]: #0,1
                this_var_name = dct[key][0]
                analysis_str += ' + lag({name},{lag})'.format(name=this_var_name, lag=this_lag)            
        elif key == 'C':
            for c_i in range(len(dct[key])):
                this_confound_name = dct[key][c_i][0]
                for this_lag in dct[key][c_i][1]:
                    analysis_str += ' + lag({name}, {lag})'.format(name = this_confound_name, lag=this_lag)               

    if include_time:
        analysis_str += ' + Time'
    if include_month:
        analysis_str += ' + C(Month)'

    if verbose:
        print(analysis_str)  

    mod = smf.ols(formula = analysis_str, data = df)

    if fit_method == 'OLS':
        res = mod.fit()
    if fit_method == 'OLS-NW':
        res = mod.fit(cov_type='HAC', cov_kwds={'maxlags':5, 'use_correction': True})

    x_pval = None
    x_beta = None
    intvn_var = None
    x_se = None
    
    if 'X' in dct:
        intvn_var = "lag({x}, {l})".format(x=dct['X'][0], l=trt_lag)
        x_pval = res.pvalues[intvn_var]
        x_beta = res.params[intvn_var]
        x_se = res.bse[intvn_var]
    
    return res, intvn_var, x_pval, x_beta, x_se

def run_gls(df, dct, fit_method, trt_lag, include_time=True, include_month=True, verbose=False):
    # Add in intercept
    df['Intercept'] = 1

    # Create exogenous variables
    exog_names = ['Intercept']

    for key in dct:
        if key in ['X','Y']:
            for this_lag in dct[key][1]: #0,1
                this_var_name = dct[key][0]
                lagged_name = this_var_name + '_lag' + str(this_lag)
                df[lagged_name] = df[this_var_name].shift(this_lag)
                exog_names.append(lagged_name)
        elif key == 'C':
            for c_i in range(len(dct[key])):
                this_confound_name = dct[key][c_i][0]
                for this_lag in dct[key][c_i][1]:
                    lagged_name = this_confound_name + '_lag' + str(this_lag)
                    df[lagged_name] = df[this_confound_name].shift(this_lag)
                    exog_names.append(lagged_name)        


    if include_month:
        month_dummies = []
        month_vals = sorted(df['Month'].unique())
        for val_idx in range(len(month_vals)):
            if val_idx == 0:
                continue
            this_val = month_vals[val_idx]
            df['Month_Dummy' + str(this_val)] = (df['Month'] == this_val).astype(float)
            month_dummies.append('Month_Dummy' + str(this_val))
        exog_names += month_dummies

    if include_time:
        exog_names.append('Time')

    if verbose:
        print('endog: ', dct['Y'][0], 'exog: ', exog_names)
        # correlation:
        #if PS:
        #    print(this_df[exog_names].corr()[['PS']])

    if fit_method == 'GLSAR':
        #print([dct['Y'][0]] + exog_names)
        #df.to_csv('df.csv')
        #print('subset:', [dct['Y'][0]] + exog_names)
        df_dropna = df.dropna(subset=[dct['Y'][0]] + exog_names)

        if verbose:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            """
            for i in range(len(exog_names)):
                if (exog_names[i] == 'Intercept') or (exog_names[i].startswith('Month')):
                    continue
                print(exog_names[i])
                df_dropna[exog_names[i]] = df_dropna[exog_names[i]].astype(float)
                print('vif', variance_inflation_factor(df_dropna[exog_names].values, i))
            """
            corrs = df_dropna[exog_names].corr()
            for i in range(len(exog_names)):
                if 'PS' not in exog_names[i]:
                    continue
                this_corrs = corrs.iloc[i, :]
                print('var, sorted corrs: ', this_corrs.sort_values())
        
        #print('df_dropna')
        #df_dropna.to_csv('df_dropna_new.csv')
        
        mod = sm.GLSAR(endog=df_dropna[dct['Y'][0]], exog = df_dropna[exog_names], rho=1)

        res = mod.iterative_fit(50)

        #print('glsar rho: ', mod.rho)

    else:
        # GLS
        df_dropna = df.dropna()

        # First run OLS
        first_ols_mod = sm.OLS(endog=df_dropna[dct['Y'][0]], exog = df_dropna[exog_names])
        first_ols_res = first_ols_mod.fit()

        # Then run an OLS on the residuals
        ols_res_df = pd.DataFrame({'res': first_ols_res.resid})
        resid_fit = smf.ols('res ~ lag(res, 1)', data=ols_res_df).fit()

        # Construct sigma
        rho = resid_fit.params[1]
        #print(rho)
        from scipy.linalg import toeplitz
        order = toeplitz(np.arange(df_dropna.shape[0]))
        sigma = rho ** order

        # Then GLS
        mod = sm.GLS(endog=df_dropna[dct['Y'][0]], exog = df_dropna[exog_names], sigma=sigma)

        res = mod.fit()
        
    x_pval = None
    x_beta = None
    x_se = None
    intvn_var = None
    
    if 'X' in dct:
        intvn_var = "{x}_lag{l}".format(x=dct['X'][0],l=trt_lag)
        x_pval = res.pvalues[intvn_var]
        x_beta = res.params[intvn_var]
        x_se = res.bse[intvn_var]
    
    return res, intvn_var, x_pval, x_beta, x_se

def normalize_df(this_df, dct):
    if 'X' in dct:
        this_df[dct['X'][0]] = (this_df[dct['X'][0]] - this_df[dct['X'][0]].mean())/this_df[dct['X'][0]].std() 
    this_df[dct['Y'][0]] = (this_df[dct['Y'][0]] - this_df[dct['Y'][0]].mean())/this_df[dct['Y'][0]].std()
    if 'C' in dct:
        for c_i in range(len(dct['C'])):
            #if 'Month' in dct['C'][c_i][0]: #'Time' in dct['C'][c_i][0] or 
            #    continue
            #else:
            this_df[dct['C'][c_i][0]] = (this_df[dct['C'][c_i][0]] - this_df[dct['C'][c_i][0]].mean())/this_df[dct['C'][c_i][0]].std()   
            
    return this_df

# Update: specify exact input to lag on the RHS. LHS is Y
# Example dct: {'X': ('tgc', [1,2]), 'Y': ('vegan_recipes', [2]), 'C': [('okja', [2]), ('recipes', [2])]}
# result: vegan_recipes ~ lag(tgc,1) + lag(tgc,2) + lag(vegan_recipes, 2) + lag(okja, 2) + lag(recipes,2)

def run_analysis(dct, df, trt_lag, non_add_L_set, 
                 PS=False,PS_logistic=False,
                 add_L=True, 
                 include_time=True, include_month=True, fit_method='GLS', 
                 difference=False, normalize=False, verbose=False):
    
    assert fit_method in ['OLS', 'OLS-NW', 'GLSAR', 'GLS']
        
    this_df = df.copy()
    
    #print('PS: ', PS)
    
    assert 'Y' in dct
  
    if difference:
        if 'X' in dct:
            this_df[dct['X'][0]] = this_df[dct['X'][0]].diff()
            #print('X value counts after differencing: ', this_df[dct['X'][0]].value_counts())
        this_df[dct['Y'][0]] = this_df[dct['Y'][0]].diff()
        if 'C' in dct:
            for c_i in range(len(dct['C'])):
                this_df[dct['C'][c_i][0]] = this_df[dct['C'][c_i][0]].diff()

    #print('Time: ', this_df['Time'].value_counts())
                
    # Normalize before PS score if not logistic
    if normalize:
        this_df = normalize_df(this_df, dct)      

    # Add PS
    if PS:
        this_df, dct = add_ps(this_df, dct, trt_lag, add_L, non_add_L_set, 
                              include_time, PS_logistic, include_month, verbose)    
            
    if fit_method in ['OLS', 'OLS-NW']:
        res, intvn_var, x_pval, x_beta, x_se = run_ols(this_df, dct, fit_method, trt_lag, 
                                                 include_time, include_month, verbose)
    
    # Fit method is GLSAR
    elif fit_method in ['GLS', 'GLSAR']:
        res, intvn_var, x_pval, x_beta, x_se = run_gls(this_df, dct, fit_method, trt_lag, 
                                                 include_time, include_month, verbose)
    
    if verbose:
        print(res.summary())
        adf_pval = adf_test(res.resid)
        kpss_pval = kpss_test(res.resid)
        print('adf_pval: ', adf_pval)
        print('kpss_pval: ', kpss_pval)
        if adf_pval > 0.05 or kpss_pval < 0.05:
            print('FAILED')
        if adf_pval > 0.05 and kpss_pval < 0.05:
            print('FAILED BOTH')    
    #print('intvn_var: ', intvn_var)
    #print('d: ', x_beta/np.std(res.resid))
    
    return x_pval, x_beta, x_se, res

# Run controls

"""
Positive and negative controls.
Positive controls: (sum of TS albums) x each TS outcome
Negative controls: 
1) (each TS album + sum) x vegan recipes
2) (each doc + sum) x TS outcomes

Binary approach: 
Y_t ~ I_t + L_t + L_t-1 + Y_t-1
"""


    

def run_controls(df, pval_thresh=0.05, PS=False, PS_logistic=False, fit_method='OLS', model='lagged', bonf_denom=1,
                add_L=True, difference=False, normalize=False, include_month=True, include_time=True,
                verbose=False):
    assert model in ['assoc', 'lagged', 'contemp']
    if model == 'assoc':
        assert add_L is False

    # Positive controls #1
    pval_thresh = 0.05/bonf_denom
    total = 0
    passed = 0
    
    trt_lag = 1 if (model == 'lagged') else 0
    non_add_L_set = []
    confounds = []
    
    for X in ['all_ts_albums']:
        for Y in all_ts_outcomes:
            df, analysis_dct = prepare_analysis_dct(df.copy(), X, Y, 
                                                    trt_lag, 
                                                    model, 
                                                    add_L, include_time, 
                                                    include_month, 
                                                    confounds, non_add_L_set)
            
            x_pval, x_beta, _, _ = run_analysis(analysis_dct, df.copy(), trt_lag, non_add_L_set,
                                                PS=PS, PS_logistic=PS_logistic,
                                          fit_method=fit_method,
                                         difference=difference, normalize=normalize,
                                               verbose=verbose, include_time=False,
                                               include_month=False)

            #print(X, Y, x_pval)
            if x_pval < pval_thresh:
                passed += 1
            total += 1
    print('TS Pos Control tests success fraction: ', float(passed)/total, 'Passed: ', passed, 'Total: ', total)
 
    # Positive controls #2
    total = 0
    passed = 0
    
    for X in ['all_drake_albums']:
        for Y in all_drake_outcomes:
            df, analysis_dct = prepare_analysis_dct(df.copy(), X, Y, 
                                                    trt_lag, 
                                                    model, 
                                                    add_L, include_time, 
                                                    include_month, 
                                                    confounds, non_add_L_set)                
            #for alb in all_ts_albums:
            #    analysis_dct['C'].append((alb, 2))
            x_pval, x_beta, _, _ = run_analysis(analysis_dct, df.copy(), trt_lag, non_add_L_set,
                                                PS=PS,
                                          PS_logistic=PS_logistic,
                                          fit_method=fit_method,
                                         difference=difference, normalize=normalize,
                                               verbose=verbose, include_time=False,
                                               include_month=False)

            #print(X, Y, x_pval)

            if x_pval < pval_thresh:
                passed += 1
            total += 1
    print('Drake Pos Control tests success fraction: ', float(passed)/total, 'Passed: ', passed, 'Total: ', total)


    # Negative controls #1
    total1 = 0
    passed1 = 0
    
    X_vals = all_ts_albums + ['all_ts_albums'] + all_drake_albums + ['all_drake_albums']
    confounds = ['health', 'animal_welfare', 'all_climate']
    for X in X_vals:
        for Y in all_primary_outcomes:
            df, analysis_dct = prepare_analysis_dct(df.copy(), X, Y, 
                                                    trt_lag, 
                                                    model, 
                                                    add_L, include_time, 
                                                    include_month, 
                                                    confounds, non_add_L_set)  
                   
            x_pval, x_beta, _, _ = run_analysis(analysis_dct, df.copy(), trt_lag, 
                                                non_add_L_set,
                                                PS=PS, 
                                          PS_logistic=PS_logistic,
                                          fit_method=fit_method,
                                         difference=difference, normalize=normalize,
                                               verbose=verbose, include_time=False,
                                               include_month=False)
            if x_pval >= pval_thresh:
                passed1 += 1
            else:
                print('failed')
                print(X, Y, x_pval)
            total1 += 1
    print('Neg Control #1 tests success fraction: ', float(passed1)/total1, 'Passed: ', passed1, 'Total: ', total1)
    
    # Negative controls #2
    # Add in the other docs as precision covariates?
    total2 = 0
    passed2 = 0
    
    
    Y_vals = all_ts_outcomes + all_drake_outcomes
    for X in all_docs + ['all_docs']:
        for Y in Y_vals:
            
            df, analysis_dct = prepare_analysis_dct(df.copy(), X, Y, 
                                                    trt_lag, 
                                                    model, 
                                                    add_L, include_time, 
                                                    include_month, 
                                                    confounds, non_add_L_set)  
            
            x_pval, x_beta, _, _ = run_analysis(analysis_dct, df.copy(), trt_lag, 
                                                non_add_L_set,
                                                PS=PS, 
                                          PS_logistic=PS_logistic,
                                          fit_method=fit_method,
                                         difference=difference, normalize=normalize,
                                               verbose=verbose, include_time=False,
                                               include_month=False)

            if x_pval >= pval_thresh:
                passed2 += 1
            else:
                print('failed')
                print(X, Y, x_pval)
            total2 += 1
            
    print('Neg Control #2 success fraction: ', float(passed2)/total2, 'Passed: ',  passed2, 'Total: ', total2)
    print('Neg Control success fraction: ', float(passed1 + passed2)/(total1 + total2), 'Passed: ',  passed1 + passed2, 'Total: ', total1 + total2)

"""
Models: assoc, lagged, contemp, bin.
Assoc: Y_t ~ X_t
Contemp: Y_t ~ PS_t + X_t + X_t-1 + Y_t-1 + L_t + L_t-1 + Month
Lagged: Y_t ~ PS_t-1 + X_t-1 + X_t-2 + L_t-1 + L_t-2 + Y_t-2 + Month
Bin: Y_t ~ PS + I_t + I_t*Time + Time + L_t + Month #+ Y_t-1 
"""
def run_primary_analyses(df, X, outcome, PS=False, PS_logistic=False, fit_method='OLS', model='lagged', 
                         add_L=True, difference=False, include_time=False, 
                         include_month=False, duration_months=None, normalize=False, verbose=False):
    assert model in ['assoc', 'lagged', 'contemp', 'bin']
    if model is 'assoc':
        assert add_L is False
        assert PS is False

    if units == 'months':
        other_docs = ['fok', 'cowspiracy', 'owth', 'tgc']

    elif units == 'weeks':
        other_docs = ['fok', 'cowspiracy', 'okja', 'wth', 'tgc']
    elif units == 'days':
        other_docs = ['fok', 'cowspiracy', 'okja', 'wth']
    else:
        assert 0
    bin_other_docs = {'fok': [], 'cowspiracy': ['fok'], 'owth': ['fok', 'cowspiracy'],
                      'okja': ['fok', 'cowspiracy'], 'wth': ['fok', 'cowspiracy'],
                         'tgc': ['fok', 'cowspiracy', 'owth'],
                     'yawye': ['fok', 'cowspiracy', 'owth', 'tgc']}
        
    if duration_months is not None:
        assert model == 'bin'
        init_date = df['ds'][0]
        this_date = '{y}-{m}-{d}'.format(y=netflix_release_dates[X][0], 
                                             m=netflix_release_dates[X][1], d=netflix_release_dates[X][2])

        release_date_time = date_difference(init_date, this_date, units)
        
        if units == 'months':
            offset = duration_months
        elif units == 'weeks':
            offset = duration_months*4
        else:
            assert 0
        df = df.loc[df['Time'] <= release_date_time + offset]
        
        print('Intvn start time: ', df.loc[df[X + '_bin'] == 1].head()['ds'])
        print('After truncating by duration_months: ', df.tail()['ds'])
        
    """
    if combine_owth:
        disaggr_docs = list(set(all_docs) - set(['okja', 'wth'])) + ['owth']
    else:
        disaggr_docs = list(set(all_docs)) 
    """
        
    trt_lag = 1 if (model == 'lagged') else 0
    #trt_lag = 2 if (model == 'lagged') else 0
    
    confounds = ['health', 'animal_welfare', 'all_climate']
    trends_full_additional_confounds = ['food']
    trends_inf_additional_confounds = ['informative_bare']
    trends_behav_additional_confounds = ['behavior_bare']
    consumption_additional_confounds = ['rdpi']
    non_add_L_set = []

    
    category_dct = {'vegan': 'trends_full', 'vegetarian': 'trends_full', 
                    'plant_based_plus_plant_based': 'trends_full', 
                    'vegan_informative': 'trends_inf', 'vegetarian_informative': 'trends_inf',
       'plant_based_informative': 'trends_inf', 
                    'vegan_behavior': 'trends_behav', 'vegetarian_behavior': 'trends_behav',
       'plant_based_behavior': 'trends_behav',
                       'ksu_chicken': 'consumption', 'ksu_pork': 'consumption', 
                    'ksu_beef': 'consumption', 'Zhao': 'consumption', 'NeuhoferLusk': 'consumption',
                   'StewartMilk': 'consumption', 'StewartPBMilk': 'consumption'}
    
    if category_dct[outcome] == 'trends_full':
        confounds += trends_full_additional_confounds
    elif category_dct[outcome] == 'trends_inf':
        confounds += trends_inf_additional_confounds
    elif category_dct[outcome] == 'trends_behav':
        confounds += trends_behav_additional_confounds        
    else:
        assert category_dct[outcome] == 'consumption'
        confounds += consumption_additional_confounds
            
    #for X in disaggr_docs + ['all_docs']:
    #, 
    
    analysis_dct = {'C': [], 'Ind_PS': []}
    
    if include_time:
        #df['Time**2'] = df['Time']*df['Time']
        analysis_dct['C'].append(('Time', [0]))
        analysis_dct['Ind_PS'].append(('Time', [0]))
        #analysis_dct['C'].append(('Time**2', [0]))
        #analysis_dct['Ind_PS'].append(('Time**2', [0]))
        
    if include_month:
        month_dummies = []
        month_vals = sorted(df['Month'].unique())
        for val_idx in range(len(month_vals)):
            if val_idx == 0:
                continue
            this_val = month_vals[val_idx]
            df['Month_Dummy' + str(this_val)] = (df['Month'] == this_val).astype(float)
            month_dummies.append('Month_Dummy' + str(this_val))
        
        
            analysis_dct['C'].append(('Month_Dummy' + str(this_val), [0]))
            analysis_dct['Ind_PS'].append(('Month_Dummy' + str(this_val), [0]))
    
    
    if model == 'lagged':
        #"""
        analysis_dct['X'] = (X, [trt_lag,trt_lag+1])
        analysis_dct['Y'] = (outcome, [trt_lag+1])
         
        
        for c in confounds:
            if add_L and c not in non_add_L_set:
                analysis_dct['C'].append((c, [trt_lag,trt_lag+1]))
            else:
                analysis_dct['C'].append((c, [trt_lag+1]))
                
                
        analysis_dct['Dep_PS'] = (X, [0])
        analysis_dct['Ind_PS'] += [(X, [1]), (outcome, [1])]

        for c in confounds:
            analysis_dct['Ind_PS'].append((c, [1]))          
            if add_L:
                analysis_dct['Ind_PS'].append((c, [0]))   

    elif model == 'contemp':
        analysis_dct['X'] = (X, [0,1])
        analysis_dct['Y'] = (outcome, [1]) 
        
        for c in confounds:
            if add_L and c not in non_add_L_set:
                analysis_dct['C'].append((c, [0,1]))
            else:
                analysis_dct['C'].append((c, [1]))  
                
        analysis_dct['Dep_PS'] = (X, [0])
        analysis_dct['Ind_PS'] += [(X, [1]), (outcome, [1])]

        for c in confounds:
            analysis_dct['Ind_PS'].append((c, [1]))          
            if add_L:
                analysis_dct['Ind_PS'].append((c, [0]))                 


    elif model == 'assoc':
        analysis_dct['X'] = (X, [0])
        analysis_dct['Y'] = (outcome, [])   
        
    elif model == 'bin':
        if not bin_slope:
            #(X + '_bin*Time', [0])
            analysis_dct['X'] = (X + '_bin', [0])
            analysis_dct['Y'] = (outcome, [])
            analysis_dct['C'].append((X + '_bin*Time_C', [0]))  
            if add_L:
                for c in confounds:
                    analysis_dct['C'].append((c, [0]))
                                        
        else:
            #(X + '_bin*Time', [0])
            analysis_dct['X'] = (X + '_bin*Time_C', [0])
            analysis_dct['Y'] = (outcome, [])
            analysis_dct['C'].append((X + '_bin', [0]))   
            if add_L:
                for c in confounds:
                    analysis_dct['C'].append((c, [0]))            
                
        # Specifically the PS regression
        df[X + '_bin_orig'] = df[X + '_bin'].copy()
        analysis_dct['Dep_PS'] = (X + '_bin_orig', [0])
        analysis_dct['Ind_PS'] = []
        if add_L:
            for c in confounds:
                analysis_dct['Ind_PS'].append((c, [0]))            
    
    # Add in confounds
    if X != 'all_docs':
        for other_doc in other_docs:
            if other_doc == X:
                continue
            if model == 'lagged':
                if add_L:
                    analysis_dct['C'].append((other_doc, [trt_lag,trt_lag+1]))
                else:
                    analysis_dct['C'].append((other_doc, [trt_lag+1]))                    
            elif model == 'contemp':
                if add_L:
                    analysis_dct['C'].append((other_doc, [0,1]))
                else:
                    analysis_dct['C'].append((other_doc, [1]))                
            elif model == 'bin':
                if other_doc in bin_other_docs[X]:
                    if add_L:
                        analysis_dct['C'].append((other_doc, [0]))
            elif model == 'assoc':
                pass


    #print('analysis_dct: ', analysis_dct)
            
    x_pval, x_beta, x_se, _ = run_analysis(analysis_dct, df, trt_lag, non_add_L_set, 
                                        PS=PS, PS_logistic=PS_logistic, fit_method=fit_method,
                                  add_L=add_L,
                                  difference=difference,
                                  include_time=False,
                                  include_month=False,
                                  normalize=normalize,
                                 verbose=verbose)          

    return x_pval, x_beta, x_se

def prepare_analysis_dct(df, X, Y, trt_lag, model, add_L, include_time, include_month, confounds, non_add_L_set):
    analysis_dct = {'C': [], 'Ind_PS': []}
    
    if include_time:
        #df['Time**2'] = df['Time']*df['Time']
        analysis_dct['C'].append(('Time', [0]))
        analysis_dct['Ind_PS'].append(('Time', [0]))
        #analysis_dct['C'].append(('Time**2', [0]))
        #analysis_dct['Ind_PS'].append(('Time**2', [0]))
        
    if include_month:
        month_dummies = []
        month_vals = sorted(df['Month'].unique())
        for val_idx in range(len(month_vals)):
            if val_idx == 0:
                continue
            this_val = month_vals[val_idx]
            df['Month_Dummy' + str(this_val)] = (df['Month'] == this_val).astype(float)
            month_dummies.append('Month_Dummy' + str(this_val))
        
        
            analysis_dct['C'].append(('Month_Dummy' + str(this_val), [0]))
            analysis_dct['Ind_PS'].append(('Month_Dummy' + str(this_val), [0]))
    
    
    if model == 'lagged':
        #"""
        analysis_dct['X'] = (X, [trt_lag,trt_lag+1])
        analysis_dct['Y'] = (Y, [trt_lag+1])
         
        
        for c in confounds:
            if add_L and c not in non_add_L_set:
                analysis_dct['C'].append((c, [trt_lag,trt_lag+1]))
            else:
                analysis_dct['C'].append((c, [trt_lag+1]))
                
                
        analysis_dct['Dep_PS'] = (X, [0])
        analysis_dct['Ind_PS'] += [(X, [1]), (Y, [1])]

        for c in confounds:
            analysis_dct['Ind_PS'].append((c, [1]))          
            if add_L:
                analysis_dct['Ind_PS'].append((c, [0]))   

    elif model == 'contemp':
        analysis_dct['X'] = (X, [0,1])
        analysis_dct['Y'] = (Y, [1]) 
        
        for c in confounds:
            if add_L and c not in non_add_L_set:
                analysis_dct['C'].append((c, [0,1]))
            else:
                analysis_dct['C'].append((c, [1]))  
                
        analysis_dct['Dep_PS'] = (X, [0])
        analysis_dct['Ind_PS'] += [(X, [1]), (Y, [1])]

        for c in confounds:
            analysis_dct['Ind_PS'].append((c, [1]))          
            if add_L:
                analysis_dct['Ind_PS'].append((c, [0]))                 


    elif model == 'assoc':
        analysis_dct['Dep_PS'] = None
        analysis_dct['Ind_PS'] = None
        analysis_dct['X'] = (X, [0])
        analysis_dct['Y'] = (Y, []) 
    
    else:
        assert 0
    
    return df, analysis_dct