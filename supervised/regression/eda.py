import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

def sub_boxplots(data, rows, cols):
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k,v in data.items():
        sns.boxplot(y=k, data=data, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


def sub_kde_plots(data, rows, cols):
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k,v in data.items():
        sns.kdeplot(x=k, data=data, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

def linearity_plot(df, y, x_cols, rows, cols):

    # create generate plot structure
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10))
    fig.suptitle(f'{y} Relationships with Covariates', fontsize=20)
    axes = axes.flatten()
    
    # build scatter subplots
    for i, var in enumerate(x_cols):

        scatter_plot = sns.scatterplot(data=df, x=var, y=y, marker='.', ax=axes[i])
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(y)
        axes[i].set_title(f"{y}~{var}", fontsize=12)
        axes[i].tick_params(axis='x', rotation=-35)  # Rotate x-ticks to avoid overlap
    
    # create a global legend and customize spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.show()


def calc_vif(X):

    # create VIF dataframe for covariates
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [vif(X.values, i) for i in range(len(X.columns))]
    vif_data = vif_data.sort_values(by=['VIF'], ascending=False)
    display(vif_data)
    
    return vif_data
    



def del_multico(df, vif_df, num, by='num'):
    """
    Removes multicollinear features from df based on VIF.

    Parameters:
    df     - DataFrame containing feature variables
    vif_df - DataFrame with VIF values (must have 'feature' and 'VIF' columns)
    num    - If `by='num'`, number of features to remove.
             If `by='thres'`, maximum allowed VIF value.
    by     - Removal strategy: 'num' (fixed number) or 'thres' (VIF threshold).

    Returns:
    df with selected features after removing high VIF features.
    """

    new_X = df.copy()  # Copy to avoid modifying the original dataframe

    while True:
        vif_df = calc_vif(new_X)  # Recalculate VIF
        
        # Check if all VIF values are below threshold
        if vif_df["VIF"].max() < num:
            break  # Stop if all VIFs are below threshold

        # Find feature with the highest VIF
        feature_to_remove = vif_df.sort_values("VIF", ascending=False).iloc[0]["feature"]

        # Drop feature
        new_X = new_X.drop(columns=[feature_to_remove])

    return new_X



def dist_shape(df):
    """
    purpose: calculate skewness and kurtosis for all columns in a dataset
    params:
        df (pd.DataFrame): dataset
    output: pd.DataFrame: a DataFrame containing distribution statistics for each column
    """
    
    # calculate skewness, and kurtosis
    shape = pd.DataFrame({'skewness': df.skew(), 'kurtosis': df.kurt()})
    
    return shape


def orq_transformation(df):
    """
    Applies a rank-based inverse normal transformation to all numeric columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: Transformed DataFrame with numeric columns transformed to follow a standard normal distribution.
    """
    transformed_df = df.copy()  # Create a copy to avoid modifying the original
    for col in df.columns:
        ranks = stats.rankdata(transformed_df[col])  # Compute ranks
        transformed_df[col] = stats.norm.ppf((ranks - 0.5) / len(transformed_df[col]))  # Apply inverse normal transformation

    return transformed_df