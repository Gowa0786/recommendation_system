import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def hitrate_at_5(preds, train_data):
    """Hitrate@5 для LambdaRank (feval)"""
    y_true = train_data.get_label()
    groups = train_data.get_group()
    
    hitrates = []
    ptr = 0
    
    for group_size in groups:
        if group_size >= 5:
            group_preds = preds[ptr:ptr + group_size]
            group_true = y_true[ptr:ptr + group_size]
            top_5_idx = np.argsort(-group_preds)[:5]
            hit = 1 if np.any(group_true[top_5_idx] == 1) else 0
            hitrates.append(hit)
        elif group_size > 0:
            group_true = y_true[ptr:ptr + group_size]
            hit = 1 if np.any(group_true == 1) else 0
            hitrates.append(hit)
        ptr += group_size
    
    return 'hitrate@5', np.mean(hitrates), True

def get_lambdarank_params(y_train):
    """Автоматический расчет параметров на основе данных"""
    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    
    return {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [5, 10],
        'scale_pos_weight': scale_pos_weight,
        'lambdarank_truncation_level': 10,
        'lambdarank_norm': True,
        'label_gain': [0, 1],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': max(20, int(20 * scale_pos_weight / 10)),
        'max_depth': 7,
        'lambda_l1': 0.15,
        'lambda_l2': 0.15,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'verbosity': -1,
        'seed': 42,
    }

def prepare_lambdarank_data(df, timestamp_col='timestamp', split_time=None):
    """Подготовка данных для LambdaRank: сортировка и группы"""
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    if split_time:
        df_train = df[df[timestamp_col] < split_time].copy()
        df_test = df[df[timestamp_col] >= split_time].copy()
    else:
        split_time = df[timestamp_col].quantile(0.8)
        df_train = df[df[timestamp_col] < split_time].copy()
        df_test = df[df[timestamp_col] >= split_time].copy()
    
    df_train = df_train.sort_values('user_id').reset_index(drop=True)
    df_test = df_test.sort_values('user_id').reset_index(drop=True)
    
    train_group_sizes = df_train.groupby('user_id').size().values
    test_group_sizes = df_test.groupby('user_id').size().values
    
    return df_train, df_test, train_group_sizes, test_group_sizes

def normalize_features(df_train, df_test, num_cols, exclude_cols=None):
    """StandardScaler для числовых признаков"""
    if exclude_cols is None:
        exclude_cols = ['timestamp', 'user_id', 'post_id', 'target']
    
    cols_to_norm = [c for c in num_cols if c not in exclude_cols]
    
    if cols_to_norm:
        scaler = StandardScaler()
        df_train[cols_to_norm] = scaler.fit_transform(df_train[cols_to_norm])
        df_test[cols_to_norm] = scaler.transform(df_test[cols_to_norm])
        return df_train, df_test, scaler
    
    return df_train, df_test, None

def create_lambdarank_datasets(X_train, y_train, train_group_sizes,
                              X_test, y_test, test_group_sizes,
                              categorical_cols):
    """Создание Dataset'ов для LightGBM"""
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_test[col] = X_test[col].astype('category')
    
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_group_sizes,
        categorical_feature=categorical_cols,
        free_raw_data=False
    )
    
    valid_data = lgb.Dataset(
        X_test,
        label=y_test,
        group=test_group_sizes,
        reference=train_data,
        categorical_feature=categorical_cols,
        free_raw_data=False
    )
    
    return train_data, valid_data

def train_lambdarank(train_data, valid_data, params, num_boost_round=500):
    """Обучение модели LambdaRank с hitrate@5"""
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[valid_data],
        valid_names=['valid'],
        feval=hitrate_at_5,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    return model
