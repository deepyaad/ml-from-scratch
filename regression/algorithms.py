'''
Author: Anissa Vaughn
Purpose: Code for Regression and Classification Algorithms Derived from Math Formulas
'''
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


class Regression:

    def __init__(self, feature_names):
            self.feature_names = feature_names

    def mse(self, y_true, y_pred):
        """ Compute Mean Squared Error """
        return np.mean((y_true - y_pred) ** 2)

    def r_squared(self, y_true, y_pred):
        return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

    def _standard_errors(self, X, y_true, y_pred, β_coefficients):
        """Compute standard errors and confidence intervals"""
        N, m = X.shape
        residuals = y_true - y_pred
        residual_variance = np.sum(residuals**2) / (N - m)

        # Compute covariance matrix
        X_T_X = X.T @ X
        cov_matrix = residual_variance * np.linalg.inv(X_T_X)

        # Compute standard errors
        coeff_std_errors = np.sqrt(np.diag(cov_matrix))

        # Compute confidence intervals (95% confidence level)
        z_score = stats.norm.ppf(0.975)  # 1.96 for 95% confidence interval
        conf_intervals = np.array([
            β_coefficients - z_score * coeff_std_errors,
            β_coefficients + z_score * coeff_std_errors
        ]).T

        return coeff_std_errors, conf_intervals
        

    def display_coef(self, X, y_true, y_pred, model):
        """Return coefficients with confidence intervals"""
        coeff_std_errors, conf_intervals = self._standard_errors(X, y_true, y_pred, model.β_coefficients)

        return pd.DataFrame({
            'features': self.feature_names,
            'coefficients': model.β_coefficients,
            'stderror': coeff_std_errors,
            'min': conf_intervals[:, 0],
            'max': conf_intervals[:, 1]
        })

    
    def compare_models(self, X_train, y_train, X_test, y_test, ridge_λ_values, lasso_λ_values):
        """Train & Compare OLS, Ridge, and Lasso for different λ values"""
        results = []
        coef_results = []

        # OLS Model
        ols = self.OLS(self.feature_names)
        ols.fit(X_train, y_train)
        ols_preds = ols.predict(X_test)

        ols_sklearn = LinearRegression()
        ols_sklearn.fit(X_train, y_train)
        ols_preds_sklearn = ols_sklearn.predict(X_test)
        
        results.append({
            "Model": "OLS",
            "Lambda": None,
            "MSE (Self)": self.mse(y_test, ols_preds),
            "MSE (API)": mean_squared_error(y_test, ols_preds_sklearn),
            "R² (Self)": self.r_squared(y_test, ols_preds),
            "R² (API)": r2_score(y_test, ols_preds_sklearn)
        })
        
        coef_df= self.display_coef(X_test, y_test, ols_preds, ols)
        coef_df["Model"] = "OLS"
        coef_df["Lambda"] = "None"
        coef_results.append(coef_df)


        

        # Ridge Models for different λ values
        for λ in ridge_λ_values:
            ridge = self.Ridge(self.feature_names, λ_regularization=λ)
            ridge.fit(X_train, y_train)
            ridge_preds = ridge.predict(X_test)

            # Sci-kit Learn Ridge Regression Model
            ridge_sklearn = Ridge(alpha=λ)
            ridge_sklearn.fit(X_train, y_train)
            ridge_preds_sklearn = ridge_sklearn.predict(X_test)
            
            results.append({
                "Model": "Ridge",
                "Lambda": λ,
                "MSE (Self)": self.mse(y_test, ridge_preds),
                "MSE (API)": mean_squared_error(y_test, ridge_preds_sklearn),
                "R² (Self)": self.r_squared(y_test, ridge_preds),
                "R² (API)": r2_score(y_test, ridge_preds_sklearn)
            })
            
            coef_df = self.display_coef(X_test, y_test, ridge_preds, ridge)
            coef_df["Model"] = "Ridge"
            coef_df["Lambda"] = λ
            coef_results.append(coef_df)            

        # Lasso Models for different λ values
        for λ in lasso_λ_values:
            lasso = self.Lasso(self.feature_names, λ_regularization=λ)
            lasso.fit(X_train, y_train)
            lasso_preds = lasso.predict(X_test)

            lasso_sklearn = Lasso(alpha=λ, max_iter=10000)
            lasso_sklearn.fit(X_train, y_train)
            lasso_preds_sklearn = lasso_sklearn.predict(X_test)
            
            results.append({
                "Model": "Lasso",
                "Lambda": λ,
                "MSE (Self)": self.mse(y_test, lasso_preds),
                "MSE (API)": mean_squared_error(y_test, lasso_preds_sklearn),
                "R² (Self)": self.r_squared(y_test, lasso_preds),
                "R² (API)": r2_score(y_test, lasso_preds_sklearn)
            })
            
            coef_df = self.display_coef(X_test, y_test, lasso_preds, lasso)
            coef_df["Model"] = "Lasso"
            coef_df["Lambda"] = λ
            coef_results.append(coef_df)   

        
        # Convert results to DataFrames
        results_df = pd.DataFrame(results)
        results_df['MSE Diff'] = results_df['MSE (Self)'] - results_df['MSE (API)']
        results_df['R² Diff'] = results_df['R² (Self)'] - results_df['R² (API)']
        coef_results_df = pd.concat(coef_results, ignore_index=True)

        return results_df, coef_results_df

    
    class OLS():
        
        def __init__(self, feature_names):
            
            # initialize the parent class
            self.feature_names = feature_names
            self.β_coefficients = None
    
    
        def fit(self, X, y):
            '''train the model'''
        
            # compute the inverse of the matrix (X'X) multipled by matrix (X'y) for coefficients using normal equation
            self.β_coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
    
    
        def predict(self, X):
            """ Make predictions """
            return np.dot(X, self.β_coefficients)
            
    
    class Ridge():
    
        def __init__(self, feature_names, λ_regularization=0.1):
            
            # initialize the parent class
            self.feature_names = feature_names
            self.β_coefficients = None
    
            # define lambda
            self.λ_regularization = λ_regularization
    
        def fit(self, X, y):
            """ Train the model """
    
            # compute ridge regression coefficients: β = (X'X + λI)⁻¹ X'Y
            N, m = X.shape
            identity_matrix = np.eye(m)
            ridge_term = self.λ_regularization * identity_matrix
            self.β_coefficients = np.linalg.inv((X.T @ X) + ridge_term) @ (X.T @ y)
    
        def predict(self, X):
            """ Make predictions """
            return np.dot(X, self.β_coefficients)

    
    class Lasso():
    
        def __init__(self, feature_names, η_learning_rate=0.01, λ_regularization=0.1, iterations=1000):
    
            # initialize the parent class
            self.feature_names = feature_names
            self.β_coefficients = None
    
            # define additional variables
            self.η_learning_rate = η_learning_rate
            self.λ_regularization = λ_regularization
            self.iterations = iterations
    
        def soft_threshold(self, ρ_partial_residual, λ_regularization):
            """ 
            purpose: soft-thresholding function used for coordinate descent (piece wise function)
            params:
                ρ_partial_residual (float)
                λ_regularization (float)
            output: updated j-th coefficient
            """
            
            # if ρ's magnitude is large enough, β will be shurnk slightly based on λ
            if ρ_partial_residual < -λ_regularization:
                return ρ_partial_residual + λ_regularization
                
            elif ρ_partial_residual > λ_regularization:
                return ρ_partial_residual - λ_regularization
    
            # otherwise β will be reduced to 0
            else:
                return 0
    
        def fit(self, X, y):
            """ Train the model using coordinate descent """
            # N = instances, m = features
            N, m = X.shape
    
            # intialize coefficient vector to 0
            self.β_coefficients = np.zeros(m)
            
            # apply coordinate descent
            for _ in range(self.iterations):
    
                # iterate over each feature
                for j in range(m): 
                    
                    # compute ρ : how much residuals change when β_j is adjusted
                    residual = y - (X @ self.β_coefficients) + X[:, j] * self.β_coefficients[j]
                    ρ_partial_residual = np.dot(X[:, j], residual) / N
    
                    # update coefficient using soft-thresholding
                    self.β_coefficients[j] = self.soft_threshold(ρ_partial_residual, self.λ_regularization)
    
        def predict(self, X):
            """ Make predictions """
            return np.dot(X, self.β_coefficients)