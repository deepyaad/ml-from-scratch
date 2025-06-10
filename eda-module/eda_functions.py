import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def load_and_clean(url):

    # load dataset from github
    data = pd.read_csv(url, low_memory=False)
    og_col_count = data.shape[1]

    # remove inf and nan values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data = data.dropna(axis=1)
    removal_count = og_col_count - data.shape[1]

    # get an initial sense of the data's structure and content
    print(f'\n{removal_count} columns were removed')
    display(data.head())

    return data

def data_summary(df):
    
    # separate variables by data types
    categorical = df.select_dtypes(include=['O', bool])
    numerical = df.select_dtypes(exclude=['O', bool])
    
    # calculate summary statistics for numerical data (min, quartiles, max)
    num_stats = pd.DataFrame(numerical.describe()).T       #transposed for readibility 
    display(num_stats)
    
    # calculate frequency distributions of categorical variables
    cat_cols = categorical.columns
    for col in cat_cols:
        print(df[col].value_counts(), '\n\n\n')

    return numerical, categorical


def dist_moments(var):

    # calculate metrics
    moments = pd.DataFrame(
        {'variance': var.var(),
         'skewness': var.skew(),
         'kurtosis': var.kurtosis()
        }, index = var.columns)

    return moments


def visualize_distribution(df, var):
    
    # review summary stats for dependent variable
    target = df[[var]]
    stats = pd.concat([target.describe().T, dist_moments(target)], axis=1)
    display(stats)

    # kernel density probability distribution
    plt.figure(figsize=(20, 6))
    sns.kdeplot(data=df, x=var)
    plt.xlabel(var)
    plt.title(f'Kernel Density Estimate of {var}')
    plt.show()


def cond_boxplot(df, y, x):

    # create boxplot visual
    plt.figure(figsize=(20, 6))
    sns.boxplot(data=df, x=x, y=y, color="silver", showfliers=True)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{y}~{x} Relationship")
    plt.xticks(rotation=-35)
    plt.show()

def scatter_plot(df, y, x):

    # create scatterplot visual
    plt.figure(figsize=(20, 6))
    sns.scatterplot(data=df, x=x, y=y, marker='*')
    plt.title(f"{y}~{x} Relationship")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

def plot_Y_10x(df, y, x_cols, x_desc, cat=None):

    # initialize
    legend_handles = []
    legend_labels = []

    # create generate plot structure
    fig, axes = plt.subplots(2, 5, figsize=(18, 10))
    fig.suptitle(f'{y} Relationships with {x_desc}', fontsize=20)
    axes = axes.flatten()
    
    # build scatter subplots
    for i, var in enumerate(x_cols):

        scatter_plot = sns.scatterplot(data=df, x=var, y=y, hue=cat, marker='.', ax=axes[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(y)
        axes[i].set_title(f"{y}~{var}", fontsize=12)
        axes[i].tick_params(axis='x', rotation=-35)  # Rotate x-ticks to avoid overlap
        
        # append the legend info from the first plot
        if i == 0:
            for handle, label in zip(*scatter_plot.get_legend_handles_labels()):
                legend_handles.append(handle)
                legend_labels.append(label)
    
        # turn off the legend for individual plots
        axes[i].legend().set_visible(False)
    
    # create a global legend and customize spacing
    fig.legend(legend_handles, legend_labels, loc="upper left", ncol=2, title=cat, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()

def metadata(df):
    return pd.DataFrame({'name': df.columns, 'unique': df.nunique(), 'type': df.dtypes})

def find_hev(df, y, cat_cap, x_desc, figsize):

    var_data = metadata(df)
    x_cats = list(var_data[var_data['unique'] < cat_cap].name)
    x_num = len(x_cats)

    # determine figure size and shape
    rows = round(len(x_cats) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    fig.suptitle(f'{y} Variance over {x_desc}', fontsize=30)
    axes = axes.flatten()
    
    for i, var in enumerate(x_cats):
        var = x_cats[i]
        var_type = df[var].dtype
        sns.boxplot(data=df, x=var, y=y, color="silver", showfliers=True, ax=axes[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(y)
        axes[i].set_title(f"UCity~{var} ({var_type})")
        axes[i].tick_params(axis='x', rotation=-35)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def corr_matrix(df):

    # visualize correlation coefficients
    correlation_matrix = df.corr().round(2)
    plt.figure(figsize=(18, 6))
    sns.heatmap(data=correlation_matrix, cmap='coolwarm',center=0)

def calc_vif(X):

    # create VIF dataframe for covariates
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [vif(X.values, i) for i in range(len(X.columns))]
    vif_data = vif_data.sort_values(by=['VIF'], ascending=False)
    display(vif_data)
    
    return vif_data

def del_multico(df, vif_df, num, by='num'):

    # intialize
    x_cols = set(vif_df['feature'])

    # drop specific number of covariates
    if by == 'num':
        top_corr_x = set(vif_df.head(num)['feature'])
        new_x_cols = x_cols.difference(top_corr_x)
        new_X = df[list(new_x_cols)]

    # drop covariates based on VIF value
    elif by == 'thres':
        top_corr_x = set(vif_df[vif_df['VIF'] < num]['feature'])
        new_x_cols = x_cols.difference(top_corr_x)
        new_X = df[list(new_x_cols)]

    # error
    else:
        new_X = 'ERROR'
    
    return new_X