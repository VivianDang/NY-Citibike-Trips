import pandas as pd
from dash import dash_table
import numpy as np
import plotly.express as px

###################################### Load Data ###################################
filename = 'New York Citibike Trips.zip'

df_raw = pd.read_csv(filename, header=0)
df = df_raw.copy(deep=True)
df.to_pickle('New York Citibike Trips.pkl')

# read data
df = pd.read_pickle('New York Citibike Trips.pkl') #to load .pkl to dataframe df

###################################### Date Time Variable ###################################
# handle date time variable
df['trip_duration_format'] = pd.to_datetime(df.trip_duration, unit='m')
df['trip_duration_hour'] = (lambda x: x.dt.hour)(df.trip_duration_format)
df['trip_duration_minute'] = (lambda x: x.dt.minute)(df.trip_duration_format)
df['trip_duration_second'] = (lambda x: x.dt.second)(df.trip_duration_format)

# df.trip_duration_format = df.trip_duration_format.dt.strftime('%H:%M:%S')
df.drop(columns=['trip_duration_format', 'bike_id'], inplace=True)

df.start_time = pd.to_datetime(df.start_time)
df.stop_time = pd.to_datetime(df.stop_time)

df['start_day'] = df.start_time.dt.day
df['start_hour'] = df.start_time.dt.hour
df['end_day'] = df.stop_time.dt.day
df['end_hour'] = df.stop_time.dt.hour
df['day'] = np.where((df.start_hour >=6) & (df.start_hour <= 18), 'day',  'night')


###################################### remove outliers ###################################
df_clean = df.copy()
df_clean = df_clean[df_clean.trip_duration <= 60]

df_clean.to_pickle('clean_data.pkl')


###################################### Categorical Data ###################################

df.start_station_id = df.start_station_id.astype('category')
df.end_station_id = df.end_station_id.astype('category')
df.start_station_name = df.start_station_name.astype('category')
df.end_station_name = df.end_station_name.astype('category')
df.user_type = df.user_type.astype('category')
# df.bike_id = df.bike_id.astype('category')
df.gender = df.gender.astype('category')


###################################### Functions ###################################
def check_nan(df):
    """
    The missing value checker

    :return: DataTable
    table of variable and number of missing values in variable
    """
    # missing values check
    df_nan = pd.DataFrame([[var, df[var].isna().sum()] for var in df.columns],
                          columns=['var', 'number of missing values'])
    dt_nan = dash_table.DataTable(df_nan.to_dict('records'), [{"name": i, "id": i} for i in df_nan.columns])
    return dt_nan


def show_header(df):
    """
    Show header of df

    :return: DataTable
    table of df.head()
    """
    dt_head = dash_table.DataTable(df.head().to_dict('records'), [{"name": i, "id": i} for i in df.columns])
    return dt_head


def show_describe(df):
    """

    :return:
    """
    des = df.describe().reset_index()
    dt_describe = dash_table.DataTable(des.to_dict('records'),
                                       [{"name": i, "id": i, 'type': 'numeric', 'format': {'specifier': '.2f'}} for i in des.columns])
    return dt_describe


# dict of functions for convenient
func = {'check_nan': check_nan, 'show_header': show_header, 'show_describe': show_describe,
        'boxplot': px.box, 'violinplot': px.violin, 'histogram':px.histogram, 'barplot':px.bar, 'pie':px.pie}





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statsmodels.graphics.gofplots import qqplot
    import numpy as np

    # detect outliers by boxplot
    df_duration = df[['trip_duration_hour', 'trip_duration_minute', 'trip_duration_second']]

    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    sns.boxplot(data=df.trip_duration, ax=ax[0])
    sns.boxplot(data=df_duration, ax=ax[1])
    ax[0].set_title('Bike Trip Duration Time Outlier')
    ax[1].set_title('Bike Trip Duration Split Time Outlier')
    ax[0].set_xlabel('trip duration')
    ax[1].set_xticklabels(['hour', 'minute', 'second'])
    ax[0].set_ylabel('Duration(minute)')
    ax[1].set_ylabel('Duration')
    ax[0].grid()
    ax[1].grid()
    plt.show()

    # remove outlier
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    sns.boxplot(data=df_duration[df.trip_duration_hour <= 1], ax=ax[1])
    sns.boxplot(data=df_clean.trip_duration, ax=ax[0])
    ax[0].set_title('Bike Trip Duration Time without Outlier')
    ax[1].set_title('Bike Trip Duration Split Time without Outlier')
    ax[0].set_xlabel('trip duration')
    ax[1].set_xticklabels(['hour', 'minute', 'second'])
    ax[0].set_ylabel('Duration(minute)')
    ax[1].set_ylabel('Duration')
    ax[0].grid()
    ax[1].grid()
    plt.show()

    # PCA
    from sklearn.preprocessing import StandardScaler

    # encoding categorical variables
    df_dumy = df_clean.copy()
    df_dumy = pd.get_dummies(df_dumy, columns=['user_type', 'gender'])
    features = ['age', 'trip_duration',
                'start_day', 'start_hour', 'end_day', 'end_hour',
                'user_type_Customer', 'user_type_Subscriber', 'gender_female',
                'gender_male']
    # normalization
    X = df_dumy[features].values
    X = StandardScaler().fit_transform(X)

    # PCA
    from sklearn.decomposition import PCA

    # pca = PCA(n_components='mle', svd_solver='full') # mle will determine automatically
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(X)
    X_PCA = pca.transform(X)

    print('Original Dimension', X.shape)
    print('Transformed Dimension', X_PCA.shape)

    print(f'explained variance ratio: {pca.explained_variance_ratio_}')
    print('=' * 80)
    # actually the first five can give explanation over 90%

    # n = mle
    plt.figure()
    x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_)) + 1, 1)
    plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
    plt.title('PCA Explained Variance Ratio(n_components=mle)')
    plt.xticks(x)
    plt.xlabel('number of component')
    plt.ylabel('explained ratio')
    plt.grid()
    plt.show()
    # n = 5
    plt.figure()
    x = np.arange(1, 5 + 1, 1)
    plt.plot(x, np.cumsum(pca.explained_variance_ratio_)[:5])
    plt.title('PCA Explained Variance Ratio(n_components=5)')
    plt.xticks(x)
    plt.xlabel('number of component')
    plt.ylabel('explained ratio')
    plt.grid()
    plt.show()

    ###################################### noramality test ###################################
    import scipy.stats as st
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].hist(data=df_clean, x='trip_duration', bins=50)
    ax[0].set_title('Histogram')
    ax[0].set_xlabel('trip duration')
    ax[0].set_ylabel('count')
    ax[0].grid()
    # qq-plot before transaction
    st.probplot(df_clean.trip_duration, dist=st.norm, plot=ax[1])
    ax[1].set_title('QQ-Plot')
    ax[1].grid()
    plt.tight_layout()
    plt.show()

    alpha = 0.01
    test_result = st.kstest(df_clean.trip_duration, 'norm')
    print(f'K-S test: statistics= {test_result.statistic:.4f} p-value = {test_result.pvalue:.4f}')
    if test_result.pvalue < alpha:
        print(f'K-S test: x dataset looks not Normal')
    print('=' * 80)

    ###################################### noramality transformation ###################################
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method = 'yeo-johnson', standardize = True)
    pt.fit(df_clean[['trip_duration']])
    df_norm = pd.DataFrame(pt.transform(df_clean[['trip_duration']]), columns=['trip_duration'])


    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].hist(data=df_norm, x='trip_duration', bins=50)
    ax[0].set_title('Power Transformed Histogram')
    ax[0].set_xlabel('trip duration(transformed)')
    ax[0].set_ylabel('count')
    ax[0].grid()
    # plt.show()
    # qq-plot before transaction
    st.probplot(df_norm.trip_duration, dist=st.norm, plot=ax[1])
    ax[1].set_title('Power Transformed QQ-Plot')
    ax[1].grid()
    plt.tight_layout()
    plt.show()


    ###################################### correlation matrix ###################################
    corr = df_dumy.corr()
    fig = plt.figure(figsize=(12, 12))
    sns.heatmap(corr,
                annot=True,  # numbers annotation
                fmt=".2f",  # remove the fractions of number
                cmap="YlGnBu")  # yellow green blue
    plt.title('Heatmap')
    plt.tight_layout()
    plt.show()

    ###################################### visualization ###################################
    # kde plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 6))
    sns.kdeplot(data=df_clean, x='trip_duration', hue='gender', ax=ax[0])
    sns.kdeplot(data=df_clean, x='trip_duration', hue='user_type', ax=ax[1])
    sns.kdeplot(data=df_clean, x='trip_duration', hue='day', ax=ax[2])
    ax[0].set_title('Kernel Density Estimate of Trip Duration')
    plt.tight_layout()
    plt.show()

    # line plot
    fig = plt.figure()
    sns.lineplot(data=df_clean, x='start_hour', y='trip_duration', label='start_hour', ci=None)
    sns.lineplot(data=df_clean, x='end_hour', y='trip_duration', label='end_hour', ci=None)
    plt.title('Trip Duration Through Time')
    plt.ylabel('trip duration(minute)')
    plt.xlabel('hour of day')
    plt.xticks(np.arange(0, 24, 1))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid()
    plt.tight_layout()
    plt.show()
    # bar plot
    df_ = df_clean.groupby('start_day').count().reset_index()
    df_cus = df_clean[df_clean.user_type=='Customer']
    fig = plt.figure(figsize=(12, 8))
    sns.barplot(data=df_, x='start_day', y='start_station_id',color='darkblue')
    sns.barplot(data=df_cus, x='start_day', y='start_station_id', estimator=len, ci=None, color='lightblue')
    # legend
    import matplotlib.patches as mpatches
    top_bar = mpatches.Patch(color='darkblue', label='Subscriber')
    bottom_bar = mpatches.Patch(color='lightblue', label='Customer')
    plt.title('count of trips by user type')
    plt.xlabel('day of May')
    plt.ylabel('count of trips')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.grid()
    plt.show()

    fig = plt.figure(figsize=(12, 8))
    sns.barplot(data=df_clean, x='gender', y='age', hue='user_type', estimator=np.mean, ci=None)
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Average Age by Gender and User Type')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # catplot
    fig = plt.figure(figsize=(12, 8))
    sns.catplot(data=df_clean, x='user_type', y='trip_duration', hue='gender', alpha=0.3)
    plt.title('Trip Duration by Gender and User Type')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # countplot
    fig = plt.figure()
    sns.countplot(data=df_clean, x='start_station_id')
    plt.title('Count of Visits of Station')
    plt.xlabel('station id')
    plt.xticks([])
    plt.grid()
    plt.show()

    # pie
    df_ = df_clean.groupby('gender')
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    ax[0].pie(df_.mean().trip_duration, labels=["Female", "Male"], explode=[0.01, 0.01], autopct="%.1f%%")
    ax[1].pie(df_.count().trip_duration, labels=["Female", "Male"], explode=[0.01, 0.01], autopct="%.1f%%")
    ax[0].set_title('Average Trip Duration by Gender')
    ax[1].set_title('Number of Trips by Gender')
    plt.show()


    # displot
    fig = plt.figure(figsize=(12, 6))
    sns.displot(data=df_clean,
                x='start_day',
                hue='day',
                # kind='kde',
                multiple='stack')
    plt.title('Number of Trips in each Day')
    plt.ylabel('number of trips')
    plt.xlabel('day of May')
    plt.tight_layout()
    plt.show()

    # violin plot
    fig = plt.figure(figsize=(12, 6))
    sns.violinplot(data=df_clean, x='user_type', y='trip_duration', hue='gender', split='True')
    plt.title('Violin Plot of Trip Duration by User Type and Gender')
    plt.show()

    # regression

    fig = plt.figure()
    sns.lmplot(data=df_clean, x='start_hour', y='trip_duration_minute')
    plt.title('Scatter Plot and Regression Line')
    plt.tight_layout()
    plt.show()

    # pairplot
    df_ = df_clean[['trip_duration', 'age']]
    fig = plt.figure(figsize=(12, 8))
    sns.pairplot(df_)
    plt.show()

