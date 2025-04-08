

def arrange_pandas_df(df):
    ## change df such that it is iterable over layers (rows) and then we can access columns by column names
    df = df.transpose()
    ##RangeIndex(start=0, stop=5, step=1) are the keys, instead make it an iterable list
    return df[1:]