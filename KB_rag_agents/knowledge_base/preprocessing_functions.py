from textwrap import dedent

# 전처리 기법별 코드 매핑
TECHNIQUE_CODE = {
    # Missing Values
    "drop_high_missing_columns": dedent("""
        # Drop columns with high missing values (>50%)
        def drop_high_missing_columns(df, threshold=0.5):
            missing_ratio = df.isnull().sum() / len(df)
            high_missing_cols = missing_ratio[missing_ratio > threshold].index
            df = df.drop(columns=high_missing_cols)
            print(f"Dropped {len(high_missing_cols)} columns with >{threshold*100}% missing values")
            return df
    """),
    
    "fill_categorical_mode": dedent("""
        # Fill categorical missing values with mode
        def fill_categorical_mode(df, cols):
            for col in cols:
                if df[col].isnull().sum() > 0:
                    mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(mode_val)
                    print(f"Filled {col} with mode: {mode_val}")
            return df
    """),
    
    "fill_numerical_median": dedent("""
        # Fill numerical missing values with median
        def fill_numerical_median(df, cols):
            for col in cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"Filled {col} with median: {median_val:.2f}")
            return df
    """),
    
    "fill_numerical_mean": dedent("""
        # Fill numerical missing values with mean
        def fill_numerical_mean(df, cols):
            for col in cols:
                if df[col].isnull().sum() > 0:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    print(f"Filled {col} with mean: {mean_val:.2f}")
            return df
    """),
    
    "advanced_imputation": dedent("""
        # Advanced imputation using KNN
        from sklearn.impute import KNNImputer
        
        def advanced_imputation(df, cols, n_neighbors=5):
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[cols] = imputer.fit_transform(df[cols])
            print(f"Applied KNN imputation for {len(cols)} numerical columns")
            return df
    """),
    
    "add_missing_indicator": dedent("""
        # Add missing indicator columns
        def add_missing_indicator(df, cols):
            for col in cols:
                df[f'{col}_missing'] = df[col].isnull().astype(int)
            print(f"Added missing indicators for {len(cols)} columns")
            return df
    """),
    
    "groupwise_imputation": dedent("""
        # Group-wise imputation by median or mean
        def groupwise_imputation(df, group_col, target_cols, strategy='median'):
            for col in target_cols:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda x: x.fillna(x.median() if strategy=='median' else x.mean())
                )
            print(f"Applied {strategy} imputation by {group_col} for {len(target_cols)} columns")
            return df
    """),

    # Outliers
    "iqr_outlier_detection": dedent("""
        # IQR-based outlier detection and capping
        def iqr_outlier_detection(df, cols, factor=1.5):
            for col in cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    print(f"Capped {outlier_count} outliers in {col}")
            return df
    """),
    
    "zscore_outlier_detection": dedent("""
        # Z-score based outlier detection
        from scipy import stats
        import numpy as np
        
        def zscore_outlier_detection(df, cols, threshold=3):
            for col in cols:
                z_scores = np.abs(stats.zscore(df[col]))
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    df.loc[z_scores > threshold, col] = mean_val + threshold * std_val * np.sign(df.loc[z_scores > threshold, col] - mean_val)
                    print(f"Capped {outlier_count} outliers in {col} using Z-score")
            return df
    """),
    
    "isolation_forest_outliers": dedent("""
        # Isolation Forest for multivariate outlier detection
        from sklearn.ensemble import IsolationForest
        
        def isolation_forest_outliers(df, cols, contamination=0.1):
            if len(cols) > 1:
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outlier_labels = iso_forest.fit_predict(df[cols])
                outlier_count = (outlier_labels == -1).sum()
                
                if outlier_count > 0:
                    df = df[outlier_labels != -1]
                    print(f"Removed {outlier_count} outliers using Isolation Forest")
            return df
    """),
    
    "winsorization": dedent("""
        # Winsorization (clipping) of extreme values
        from scipy.stats import mstats
        
        def winsorization(df, cols, limits=[0.01, 0.99]):
            for col in cols:
                df[col] = mstats.winsorize(df[col], limits=limits)
            print(f"Applied winsorization to {len(cols)} columns")
            return df
    """),
    
    "power_transform": dedent("""
        # Power transform (Yeo-Johnson or Box-Cox)
        from sklearn.preprocessing import PowerTransformer
        
        def power_transform(df, cols, method='yeo-johnson'):
            pt = PowerTransformer(method=method)
            df[cols] = pt.fit_transform(df[cols])
            print(f"Applied {method} power transform to {len(cols)} columns")
            return df
    """),

    # Categorical Encoding
    "label_encoding": dedent("""
        # Label Encoding for categorical variables
        from sklearn.preprocessing import LabelEncoder
        
        def label_encoding(df, cols):
            for col in cols:
                if df[col].nunique() <= 10:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    print(f"Applied Label Encoding to {col}")
            return df
    """),
    
    "onehot_encoding": dedent("""
        # One-Hot Encoding for categorical variables
        import pandas as pd
        
        def onehot_encoding(df, cols):
            for col in cols:
                if df[col].nunique() > 10 and df[col].nunique() <= 20:
                    df = pd.get_dummies(df, columns=[col], prefix=col)
                    print(f"Applied One-Hot Encoding to {col}")
            return df
    """),
    
    "frequency_encoding": dedent("""
        # Frequency Encoding for categorical variables
        def frequency_encoding(df, cols):
            for col in cols:
                if df[col].nunique() > 20:
                    freq_map = df[col].value_counts().to_dict()
                    df[f"{col}_frequency"] = df[col].map(freq_map)
                    print(f"Applied Frequency Encoding to {col}")
            return df
    """),
    
    "ordinal_encoding": dedent("""
        # Ordinal Encoding
        from sklearn.preprocessing import OrdinalEncoder
        
        def ordinal_encoding(df, cols):
            oe = OrdinalEncoder()
            df[cols] = oe.fit_transform(df[cols].astype(str))
            print(f"Applied Ordinal Encoding to {len(cols)} columns")
            return df
    """),
    
    "target_encoding": dedent("""
        # Target Encoding
        def target_encoding(df, cols, target_col):
            for col in cols:
                target_means = df.groupby(col)[target_col].mean()
                df[f"{col}_target_encoded"] = df[col].map(target_means)
                print(f"Applied Target Encoding to {col}")
            return df
    """),
    
    "feature_hashing": dedent("""
        # Feature Hashing
        from sklearn.feature_extraction import FeatureHasher
        import pandas as pd
        
        def feature_hashing(df, cols, n_features=10):
            for col in cols:
                hasher = FeatureHasher(n_features=n_features, input_type='string')
                hashed = hasher.transform(df[col].astype(str))
                df_hashed = pd.DataFrame(hashed.toarray(), index=df.index, 
                                       columns=[f"{col}_hash_{i}" for i in range(n_features)])
                df = pd.concat([df, df_hashed], axis=1)
                print(f"Applied Feature Hashing to {col}")
            return df
    """),

    # Scaling
    "standard_scaling": dedent("""
        # Standard Scaling for numerical variables
        from sklearn.preprocessing import StandardScaler
        
        def standard_scaling(df, cols):
            scaler = StandardScaler()
            df[cols] = scaler.fit_transform(df[cols])
            print(f"Applied Standard Scaling to {len(cols)} numerical columns")
            return df
    """),
    
    "minmax_scaling": dedent("""
        # MinMax Scaling for numerical variables
        from sklearn.preprocessing import MinMaxScaler
        
        def minmax_scaling(df, cols):
            scaler = MinMaxScaler()
            df[cols] = scaler.fit_transform(df[cols])
            print(f"Applied MinMax Scaling to {len(cols)} numerical columns")
            return df
    """),
    
    "robust_scaling": dedent("""
        # Robust Scaling for numerical variables with outliers
        from sklearn.preprocessing import RobustScaler
        
        def robust_scaling(df, cols):
            scaler = RobustScaler()
            df[cols] = scaler.fit_transform(df[cols])
            print(f"Applied Robust Scaling to {len(cols)} numerical columns")
            return df
    """),
    
    "quantile_transform": dedent("""
        # Quantile Transformation
        from sklearn.preprocessing import QuantileTransformer
        
        def quantile_transform(df, cols, output_distribution='uniform'):
            qt = QuantileTransformer(output_distribution=output_distribution)
            df[cols] = qt.fit_transform(df[cols])
            print(f"Applied Quantile Transform to {len(cols)} columns")
            return df
    """),

    # Feature Selection
    "variance_threshold": dedent("""
        # Remove low variance features
        from sklearn.feature_selection import VarianceThreshold
        
        def variance_threshold(df, cols, threshold=0.01):
            selector = VarianceThreshold(threshold=threshold)
            df_selected = pd.DataFrame(
                selector.fit_transform(df[cols]),
                columns=df[cols].columns[selector.get_support()],
                index=df.index
            )
            removed_features = len(cols) - df_selected.shape[1]
            print(f"Removed {removed_features} low variance features")
            return df_selected
    """),
    
    "correlation_filter": dedent("""
        # Remove highly correlated features
        import numpy as np
        
        def correlation_filter(df, cols, threshold=0.8):
            if len(cols) > 1:
                corr_matrix = df[cols].corr().abs()
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                
                to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
                df = df.drop(columns=to_drop)
                print(f"Removed {len(to_drop)} highly correlated features")
            return df
    """),
    
    "recursive_feature_elimination": dedent("""
        # Recursive Feature Elimination
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        
        def recursive_feature_elimination(df, cols, target_col, n_features=10):
            X = df[cols]
            y = df[target_col]
            
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            rfe = RFE(estimator=estimator, n_features_to_select=n_features)
            X_rfe = rfe.fit_transform(X, y)
            
            selected_features = X.columns[rfe.support_].tolist()
            print(f"Selected {len(selected_features)} features using RFE")
            return df[selected_features + [target_col]]
    """),

    # Feature Engineering
    "polynomial_features": dedent("""
        # Create polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        
        def polynomial_features(df, cols, degree=2):
            if len(cols) >= 2:
                poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
                poly_features = poly.fit_transform(df[cols])
                feature_names = poly.get_feature_names_out(cols)
                
                poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                interaction_cols = [col for col in feature_names if ' ' in col]
                df = pd.concat([df, poly_df[interaction_cols]], axis=1)
                print(f"Created {len(interaction_cols)} polynomial interaction features")
            return df
    """),
    
    "datetime_features": dedent("""
        # Extract datetime features
        import pandas as pd
        
        def datetime_features(df, datetime_cols):
            for col in datetime_cols:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        continue
                
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                print(f"Extracted datetime features from {col}")
            return df
    """),
    
    "binning_features": dedent("""
        # Create binning features for numerical variables
        def binning_features(df, cols, bins=5):
            for col in cols:
                try:
                    df[f"{col}_binned"] = pd.cut(df[col], bins=bins, labels=False)
                    print(f"Created binned feature for {col}")
                except:
                    continue
            return df
    """),
    
    "feature_domain_interaction": dedent("""
        # Create domain-specific feature interactions
        def feature_domain_interaction(df, col1, col2, operation='multiply'):
            if operation == 'multiply':
                df[f"{col1}_{col2}_multiply"] = df[col1] * df[col2]
            elif operation == 'divide':
                df[f"{col1}_{col2}_divide"] = df[col1] / (df[col2] + 1e-8)
            elif operation == 'difference':
                df[f"{col1}_{col2}_diff"] = df[col1] - df[col2]
            print(f"Created {operation} interaction between {col1} and {col2}")
            return df
    """),
    
    "cyclical_encoding": dedent("""
        # Cyclical encoding for time features
        import numpy as np
        
        def cyclical_encoding(df, cols):
            for col in cols:
                df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / df[col].max())
                df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / df[col].max())
                print(f"Applied cyclical encoding to {col}")
            return df
    """),
    
    "rare_category_grouping": dedent("""
        # Group rare categories
        def rare_category_grouping(df, cols, threshold=0.01):
            for col in cols:
                value_counts = df[col].value_counts(normalize=True)
                rare_categories = value_counts[value_counts < threshold].index
                df[col] = df[col].replace(rare_categories, 'Other')
                print(f"Grouped {len(rare_categories)} rare categories in {col}")
            return df
    """),

    # Dimensionality Reduction
    "pca_reduction": dedent("""
        # PCA for dimensionality reduction
        from sklearn.decomposition import PCA
        
        def pca_reduction(df, cols, n_components=10):
            n_components = min(len(cols), n_components)
            pca = PCA(n_components=n_components, random_state=42)
            pca_features = pca.fit_transform(df[cols])
            
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f"PC_{i+1}" for i in range(n_components)],
                index=df.index
            )
            
            explained_variance = sum(pca.explained_variance_ratio_)
            print(f"Applied PCA: {n_components} components, {explained_variance:.3f} variance explained")
            return pca_df
    """),
    
    "tsne_reduction": dedent("""
        # t-SNE for dimensionality reduction
        from sklearn.manifold import TSNE
        
        def tsne_reduction(df, cols, n_components=2):
            n_components = min(len(cols), n_components)
            
            if len(df) > 5000:
                sample_indices = np.random.choice(len(df), 5000, replace=False)
                sample_data = df[cols].iloc[sample_indices]
            else:
                sample_data = df[cols]
            
            tsne = TSNE(n_components=n_components, random_state=42, 
                       perplexity=min(30, len(sample_data)-1))
            tsne_features = tsne.fit_transform(sample_data)
            
            tsne_df = pd.DataFrame(
                tsne_features,
                columns=[f"tSNE_{i+1}" for i in range(n_components)],
                index=sample_data.index
            )
            print(f"Applied t-SNE: {n_components} components, {len(sample_data)} samples")
            return tsne_df
    """),
    
    "umap_reduction": dedent("""
        # UMAP for dimensionality reduction
        import umap
        
        def umap_reduction(df, cols, n_components=10):
            n_components = min(len(cols), n_components)
            
            if len(df) > 10000:
                sample_indices = np.random.choice(len(df), 10000, replace=False)
                sample_data = df[cols].iloc[sample_indices]
            else:
                sample_data = df[cols]
            
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            umap_features = reducer.fit_transform(sample_data)
            
            umap_df = pd.DataFrame(
                umap_features,
                columns=[f"UMAP_{i+1}" for i in range(n_components)],
                index=sample_data.index
            )
            print(f"Applied UMAP: {n_components} components, {len(sample_data)} samples")
            return umap_df
    """),
    
    "lda_reduction": dedent("""
        # LDA for dimensionality reduction
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        
        def lda_reduction(df, cols, target_col, n_components=None):
            if target_col in df.columns and len(cols) > 0:
                n_classes = len(df[target_col].unique())
                if n_components is None:
                    n_components = min(len(cols), n_classes - 1)
                
                lda = LinearDiscriminantAnalysis(n_components=n_components)
                lda_features = lda.fit_transform(df[cols], df[target_col])
                
                lda_df = pd.DataFrame(
                    lda_features,
                    columns=[f"LDA_{i+1}" for i in range(n_components)],
                    index=df.index
                )
                print(f"Applied LDA: {n_components} components")
                return lda_df
            return df
    """),
    
    "feature_agglomeration": dedent("""
        # Feature Agglomeration
        from sklearn.cluster import FeatureAgglomeration
        
        def feature_agglomeration(df, cols, n_clusters=10):
            n_clusters = min(len(cols), n_clusters)
            agg = FeatureAgglomeration(n_clusters=n_clusters)
            agg_features = agg.fit_transform(df[cols])
            
            agg_df = pd.DataFrame(
                agg_features,
                columns=[f"Agg_{i+1}" for i in range(n_clusters)],
                index=df.index
            )
            print(f"Applied Feature Agglomeration: {n_clusters} clusters")
            return agg_df
    """),

    # Class Imbalance
    "smote_oversampling": dedent("""
        # SMOTE for handling class imbalance
        from imblearn.over_sampling import SMOTE
        
        def smote_oversampling(df, cols, target_col):
            X = df[cols]
            y = df[target_col]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            df_resampled = X_resampled.copy()
            df_resampled[target_col] = y_resampled
            
            print(f"Applied SMOTE: {len(y_resampled)} samples (original: {len(y)})")
            return df_resampled
    """),
    
    "adasyn_oversampling": dedent("""
        # ADASYN for handling class imbalance
        from imblearn.over_sampling import ADASYN
        
        def adasyn_oversampling(df, cols, target_col):
            X = df[cols]
            y = df[target_col]
            
            adasyn = ADASYN(random_state=42)
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            df_resampled = X_resampled.copy()
            df_resampled[target_col] = y_resampled
            
            print(f"Applied ADASYN: {len(y_resampled)} samples (original: {len(y)})")
            return df_resampled
    """),
    
    "random_undersampling": dedent("""
        # Random Undersampling for handling class imbalance
        from imblearn.under_sampling import RandomUnderSampler
        
        def random_undersampling(df, cols, target_col):
            X = df[cols]
            y = df[target_col]
            
            rus = RandomUnderSampler(random_state=42)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            
            df_resampled = X_resampled.copy()
            df_resampled[target_col] = y_resampled
            
            print(f"Applied Random Undersampling: {len(y_resampled)} samples (original: {len(y)})")
            return df_resampled
    """),
    
    "class_weights": dedent("""
        # Class weights for handling class imbalance
        from sklearn.utils.class_weight import compute_class_weight
        
        def class_weights(df, target_col):
            classes = df[target_col].unique()
            class_weights = compute_class_weight('balanced', classes=classes, y=df[target_col])
            
            weight_dict = dict(zip(classes, class_weights))
            print(f"Computed class weights: {weight_dict}")
            print("Use these weights in your model training")
            return weight_dict
    """)
}

def get_code_snippet(name):
    """
    Get the Python code snippet for the given preprocessing technique name.
    
    Args:
        name (str): Name of the preprocessing technique
        
    Returns:
        str: Code snippet for the technique, or None if not found
    """
    return TECHNIQUE_CODE.get(name)

def print_code_snippet(name):
    """
    Print the Python code snippet for the given preprocessing technique name.
    
    Args:
        name (str): Name of the preprocessing technique
    """
    snippet = TECHNIQUE_CODE.get(name)
    if snippet:
        print(f"--- Code for '{name}' ---")
        print(snippet)
    else:
        print(f"No snippet found for '{name}'. Available techniques:")
        for key in sorted(TECHNIQUE_CODE.keys()):
            print(f" - {key}")

def get_available_techniques():
    """
    Get list of all available preprocessing techniques.
    
    Returns:
        list: List of technique names
    """
    return sorted(TECHNIQUE_CODE.keys())

if __name__ == "__main__":
    # Example usage
    print("Available preprocessing techniques:")
    techniques = get_available_techniques()
    for i, technique in enumerate(techniques, 1):
        print(f"{i:2d}. {technique}")
    
    print("\n" + "="*50)
    print("Example: Code for median imputation")
    print_code_snippet("fill_numerical_median") 