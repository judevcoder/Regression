import pandas

X_columns_order = ["BathroomsTotal", "BedroomsTotal", "LivingArea", "LotSize", "Pool_YN", "YearBuilt"]
Y_column = ["close_price"]


def waterYNTransform(x):
    if x == 0.0:
        return 1.0
    else:
        return 0.0


def removeComma(x):
    if isinstance(x, basestring):
        x = x.replace(',', '')
    return x


def parseDict(data):
    df = pandas.DataFrame.from_dict(data, dtype='float32')
    return cleanDf(df)


def separateXY(df):
    X = df[X_columns_order].as_matrix()
    Y = df[Y_column].as_matrix()
    # Automatically determine what our X columns are
    # myXColums = list(df.columns.values);
    # myXColums = [x.encode('UTF8') for x in myXColums]
    # print "myXColums: ", myXColums;
    # try:
    #     i = myXColums.index("close_price")
    #     del myXColums[i]
    # except KeyError:
    #     pass
    #
    # X = df[myXColums].as_matrix()
    # Y = df[Y_column].as_matrix()

    return X, Y


def cleanDf(df):
    # drop if ListPrice is NaN or equal to 0
    df = df[df["close_price"].isnull() == False]
    df = df[df["close_price"] > 0]

    if "LotSize" in df:
        df = df[df["LotSize"] > 0]
    # transform "Pool_YN" to 0 (false) and 1 (true)
    if "Pool_YN" in df:
        df["Pool_YN"] = df["Pool_YN"].apply(waterYNTransform)
    # Remove Commas
    df = df.applymap(removeComma)
    # fill residual nulls with 0.0
    df = df.fillna(0.0)
    return separateXY(df)