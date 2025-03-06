from scipy import stats

def ttest_rmse(df, target):
    if target in df.index:
        target = df.index.get_loc(target)
    ttest = {}
    for col in df.index:
        ttest[col] = stats.ttest_rel(df.loc[col], df.iloc[target])
    return ttest

def ttest_rel(group1, group2):
    ttest = stats.ttest_rel(group1, group2)
    return ttest.pvalue

def anova(groups):
    f_val, p_val = stats.f_oneway(groups[0], groups[1])
    return f_val, p_val