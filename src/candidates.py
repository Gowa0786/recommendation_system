import numpy as np
import pandas as pd
from scipy import sparse
from implicit.als import AlternatingLeastSquares
import faiss
import pickle
from tqdm import tqdm
import gc
from collections import defaultdict

# ==================== ALS ====================

def create_interaction_matrix(df, user_col='user_id', item_col='post_id'):
    """Создание sparse матрицы взаимодействий для ALS."""
    users_sorted = np.sort(df[user_col].unique())
    items_sorted = np.sort(df[item_col].unique())
    
    user_to_idx = {uid: i for i, uid in enumerate(users_sorted)}
    idx_to_user = {i: uid for i, uid in enumerate(users_sorted)}
    item_to_idx = {pid: i for i, pid in enumerate(items_sorted)}
    idx_to_item = {i: pid for i, pid in enumerate(items_sorted)}
    
    rows = [user_to_idx[u] for u in df[user_col]]
    cols = [item_to_idx[p] for p in df[item_col]]
    
    matrix = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32), (rows, cols)),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    
    return matrix, user_to_idx, idx_to_user, item_to_idx, idx_to_item

def train_als_model(matrix, factors=20, iterations=15, regularization=0.01, alpha=40):
    """Обучение ALS модели."""
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        alpha=alpha,
        random_state=42,
        use_gpu=False
    )
    model.fit(matrix, show_progress=True)
    return model

def normalize_embeddings(factors):
    """L2 нормализация эмбеддингов."""
    norms = np.linalg.norm(factors, axis=1, keepdims=True)
    return factors / np.clip(norms, 1e-10, None)

def save_als_artifacts(model, user_to_idx, item_to_idx, split_time, path_prefix=''):
    """Сохранение ALS модели и маппингов."""
    user_factors = normalize_embeddings(model.user_factors)
    item_factors = normalize_embeddings(model.item_factors)
    
    np.save(f'{path_prefix}user_factors.npy', user_factors.astype(np.float32))
    np.save(f'{path_prefix}post_factors.npy', item_factors.astype(np.float32))
    
    with open(f'{path_prefix}als_mappings.pkl', 'wb') as f:
        pickle.dump({
            'user_to_idx': user_to_idx,
            'post_to_idx': item_to_idx,
            'split_time': split_time,
            'train_users_count': len(user_to_idx),
            'train_posts_count': len(item_to_idx)
        }, f)

def filter_cold_start(df_test, train_users, train_posts):
    """Исключение cold-start пользователей и постов из test."""
    return df_test[
        df_test['user_id'].isin(train_users) & 
        df_test['post_id'].isin(train_posts)
    ].copy()

def get_als_candidates(user_id, user_to_idx, model, post_idx_to_id, n=100):
    """Получение кандидатов через ALS для конкретного пользователя."""
    if user_id not in user_to_idx:
        return []
    
    user_idx = user_to_idx[user_id]
    recommendations = model.recommend(user_idx, item_users=None, N=n)
    
    candidates = []
    for item_idx, score in zip(recommendations[0], recommendations[1]):
        if item_idx in post_idx_to_id:
            candidates.append(post_idx_to_id[item_idx])
    
    return candidates[:n]

# ==================== СТАТИСТИКА (CTR) ====================

def build_user_group_stats(features_df):
    """Построение user-group статистики (CTR, просмотры, лайки)."""
    print("Building user-group statistics...")
    
    post_to_group_dict = dict(zip(features_df['post_id'], features_df['topic_code']))
    
    # User-group stats
    user_group_dict = {}
    grouped = features_df.groupby(['user_id', 'topic_code'])
    for (user_id, topic), group in grouped:
        views = len(group)
        likes = group['target'].sum()
        ctr = likes / views if views > 0 else 0.0
        user_group_dict[(int(user_id), int(topic))] = {
            'views': views,
            'likes': int(likes),
            'ctr': ctr
        }
    
    # Global group stats
    global_group_dict = {}
    global_grouped = features_df.groupby('topic_code')['target']
    for topic, targets in global_grouped:
        views = len(targets)
        likes = targets.sum()
        ctr = likes / views if views > 0 else 0.05
        global_group_dict[int(topic)] = {
            'ctr': ctr,
            'ctr_tanh': np.tanh(ctr * 3) * 0.4,
            'views': views,
            'likes': int(likes)
        }
    
    user_view_counts = features_df.groupby('user_id').size().to_dict()
    user_seen_dict = features_df.groupby('user_id')['post_id'].agg(set).to_dict()
    
    liked_df = features_df[features_df['target'] == True]
    user_liked_dict = liked_df.groupby('user_id')['post_id'].agg(set).to_dict()
    
    return {
        'user_group_dict': user_group_dict,
        'global_group_dict': global_group_dict,
        'user_view_counts': user_view_counts,
        'user_seen_dict': user_seen_dict,
        'user_liked_dict': user_liked_dict,
        'post_to_group_dict': post_to_group_dict
    }

# ==================== FAISS И ИНФЕРЕНС ====================

def build_faiss_index(item_factors, n_threads=8):
    """Построение FAISS индекса."""
    faiss.omp_set_num_threads(n_threads)
    dimension = item_factors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(item_factors.astype(np.float32))
    return index

def calculate_novelty_score(post_id, seen_set, min_post_id=0, max_post_id=10000):
    """Бонус/штраф за новизну поста."""
    if post_id in seen_set:
        return 0.8
    novelty_norm = (post_id - min_post_id) / (max_post_id - min_post_id + 1e-8)
    return 1.0 + novelty_norm * 0.2

def calculate_hybrid_score(cosine, post_id, user_id, seen_set,
                          post_to_group_dict, user_group_dict, global_group_dict):
    """Гибридный скор: Cosine + Novelty + Personal CTR + Global CTR."""
    score = cosine
    
    # Novelty
    novelty = calculate_novelty_score(post_id, seen_set)
    score *= novelty
    
    # Personal CTR
    post_group = post_to_group_dict.get(post_id, 45)
    key = (user_id, post_group)
    
    if key in user_group_dict:
        stats = user_group_dict[key]
        if stats['views'] >= 3 and stats['ctr'] > 0.1:
            ctr_bonus = min(0.3, stats['ctr'] * 0.7)
            score += ctr_bonus
    
    # Global CTR
    if post_group in global_group_dict:
        g_bonus = global_group_dict[post_group]['ctr_tanh'] * 0.05
        score += g_bonus
    
    return score

def filter_candidates_batch(post_ids, cosines, user_id, seen_set, liked_set,
                           post_to_group_dict, user_group_dict, global_group_dict,
                           min_cosine_threshold=0.6, top_n=30):
    """Фильтрация и скоринг кандидатов."""
    filtered_post_ids = []
    filtered_cosines = []
    
    for post_id, cosine in zip(post_ids, cosines):
        if post_id in liked_set:
            continue
        if post_id in seen_set and cosine < min_cosine_threshold:
            continue
        filtered_post_ids.append(post_id)
        filtered_cosines.append(cosine)
    
    if not filtered_post_ids:
        return []
    
    scores = []
    for post_id, cosine in zip(filtered_post_ids, filtered_cosines):
        score = calculate_hybrid_score(
            cosine, post_id, user_id, seen_set,
            post_to_group_dict, user_group_dict, global_group_dict
        )
        scores.append(score)
    
    scores = np.array(scores)
    if len(scores) > top_n:
        top_indices = np.argpartition(-scores, top_n)[:top_n]
        top_scores = scores[top_indices]
        sorted_top_indices = top_indices[np.argsort(-top_scores)]
        return [filtered_post_ids[idx] for idx in sorted_top_indices]
    else:
        sorted_indices = np.argsort(-scores)
        return [filtered_post_ids[idx] for idx in sorted_indices]

def process_user_batch(user_batch, user_factors, faiss_index,
                       user_id_to_idx, post_idx_to_id,
                       user_seen_dict, user_liked_dict,
                       post_to_group_dict, user_group_dict, global_group_dict,
                       k=300, top_n=30):
    """Обработка батча пользователей."""
    valid_users = []
    valid_indices = []
    
    for uid in user_batch:
        if uid in user_id_to_idx:
            valid_users.append(uid)
            valid_indices.append(user_id_to_idx[uid])
    
    if not valid_users:
        return {uid: [] for uid in user_batch}
    
    user_vectors = user_factors[valid_indices]
    D_batch, I_batch = faiss_index.search(user_vectors, k=k)
    
    batch_results = {}
    
    for i, user_id in enumerate(valid_users):
        cosines = D_batch[i]
        post_indices = I_batch[i]
        
        candidate_post_ids = []
        for idx in post_indices:
            if idx < len(post_idx_to_id):
                candidate_post_ids.append(post_idx_to_id[idx])
        
        if not candidate_post_ids:
            batch_results[user_id] = []
            continue
        
        seen_set = user_seen_dict.get(user_id, set())
        liked_set = user_liked_dict.get(user_id, set())
        
        top_posts = filter_candidates_batch(
            candidate_post_ids, cosines, user_id, seen_set, liked_set,
            post_to_group_dict, user_group_dict, global_group_dict,
            top_n=top_n
        )
        
        batch_results[user_id] = top_posts
    
    for uid in user_batch:
        if uid not in batch_results:
            batch_results[uid] = []
    
    return batch_results

# ==================== ПРОДАКШЕН ПАЙПЛАЙН ====================

def precompute_all_candidates(user_factors, post_factors,
                             user_id_to_idx, post_idx_to_id,
                             features_df=None, stats_dicts=None,
                             k=300, top_n=30, batch_size=10000,
                             save_path='precomputed_candidates.pkl'):
    """Полный пайплайн предрасчета кандидатов."""
    print("="*60)
    print("PRECOMPUTE CANDIDATES PIPELINE")
    print("="*60)
    
    if stats_dicts is None and features_df is not None:
        stats_dicts = build_user_group_stats(features_df)
    elif stats_dicts is None:
        raise ValueError("Need either features_df or stats_dicts")
    
    print("\nBuilding FAISS index...")
    faiss_index = build_faiss_index(post_factors)
    print(f"Index built: {faiss_index.ntotal} vectors")
    
    all_user_ids = np.array(list(user_id_to_idx.keys()), dtype=np.int32)
    n_users = len(all_user_ids)
    print(f"Total users: {n_users:,}")
    print(f"Using k={k}, top_n={top_n}, batch_size={batch_size}")
    
    all_results = {}
    n_batches = (n_users + batch_size - 1) // batch_size
    
    for batch_start in tqdm(range(0, n_users, batch_size), total=n_batches, desc="Processing batches"):
        batch_end = min(batch_start + batch_size, n_users)
        user_batch = all_user_ids[batch_start:batch_end]
        
        batch_results = process_user_batch(
            user_batch, user_factors, faiss_index,
            user_id_to_idx, post_idx_to_id,
            stats_dicts['user_seen_dict'],
            stats_dicts['user_liked_dict'],
            stats_dicts['post_to_group_dict'],
            stats_dicts['user_group_dict'],
            stats_dicts['global_group_dict'],
            k=k, top_n=top_n
        )
        
        all_results.update(batch_results)
        
        if batch_start % 50000 == 0 and batch_start > 0:
            gc.collect()
    
    print(f"\nSaving results to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"✓ Saved candidates for {len(all_results):,} users")
    
    result_lengths = [len(posts) for posts in all_results.values()]
    print("\n" + "="*40)
    print("RESULTS STATISTICS")
    print("="*40)
    print(f"Mean candidates per user: {np.mean(result_lengths):.1f}")
    print(f"Median: {np.median(result_lengths):.0f}")
    print(f"Min: {min(result_lengths)}")
    print(f"Max: {max(result_lengths)}")
    
    return all_results
