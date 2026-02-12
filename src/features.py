import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def create_post_embeddings(df_posts, n_components=2):
    """
    Создание TF-IDF эмбеддингов постов.
    Используется для генерации признаков постов.
    
    Параметры:
        df_posts: DataFrame с колонками ['post_id', 'text']
        n_components: количество эмбеддингов (по умолч. 2)
    """
    vectorizer = TfidfVectorizer(
        max_features=n_components,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.7
    )
    
    tfidf_matrix = vectorizer.fit_transform(df_posts['text'].astype(str))
    
    post_embeddings = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'post_embed_{i}' for i in range(n_components)]
    )
    post_embeddings['post_id'] = df_posts['post_id'].values
    
    return post_embeddings, vectorizer

def create_cyclic_features(df, timestamp_col='timestamp'):
    """
    Создание циклических признаков из времени.
    - день недели (sin/cos)
    - час дня (sin/cos)
    """
    df = df.copy()
    df['dow_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.dayofweek / 7)
    df['hour_sin'] = np.sin(2 * np.pi * df[timestamp_col].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df[timestamp_col].dt.hour / 24)
    return df

def encode_topic(df, topic_col='topic'):
    """
    Кодирование тематик постов в числовые коды.
    Возвращает df с колонкой 'topic_code' и словарь mapping'а.
    """
    df = df.copy()
    df['topic_code'] = df[topic_col].astype('category').cat.codes
    topic_mapping = dict(enumerate(df[topic_col].astype('category').cat.categories))
    return df, topic_mapping

def prepare_model_features(df, drop_cols=None):
    """
    Финальная очистка фичей перед обучением.
    Удаляет сырые колонки, оставляет закодированные.
    """
    if drop_cols is None:
        drop_cols = [
            'country', 'city', 'os', 'source', 'topic',
            'age', 'action', 'topic_covid'
        ]
    
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    return df.drop(columns=cols_to_drop)
