from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

def main(model_name):
    # 讀取玩家信息數據
    info = pd.read_csv('train_info.csv')
    
    # 獲取唯一的 player_id 列表
    unique_players = info['player_id'].unique()
    
    # 分割訓練集和測試集
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    # 讀取特徵數據
    feature_file = pd.read_csv("features.csv")
    
    # 定義目標變量欄位
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # 初始化訓練集和測試集
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)

    # 提取 unique_id 列表
    unique_id_list = [feature_file.iat[i, 0] for i in range(0, len(feature_file), 27)]
    
    # 定義特徵索引（恢復 years 特徵數）
    feature_importance = {
        "gender": [2, 3, 18, 0, 40, 5, 4, 49, 29, 15, 9, 8, 25, 21, 23, 51, 28, 14, 1, 55, 
                   24, 20, 50, 43, 17, 31, 53, 16, 10, 6, 46, 12, 26, 30, 35, 7, 22, 34, 44, 32,
                   38, 54, 19, 33, 45, 27, 11, 42, 48, 13, 41, 47, 36, 37, 39],  # 擴展至 55 個
        "hold": [4, 0, 3, 2, 5, 25, 31, 15, 45, 11, 23, 33, 40, 27, 9],
        "year": [23, 21, 3, 22, 16, 4, 10, 17, 0, 15, 25, 31, 1, 5, 9],  # 恢復 15 個特徵
        "level": [3, 0, 4, 23, 2, 1, 52, 31, 21, 5, 28, 6, 53, 15, 16, 12, 20, 22, 9, 40, 25, 17]
    }
    
    # 將特徵索引加 1
    for key in feature_importance:
        feature_importance[key] = [x + 1 for x in feature_importance[key]]
    
    # 處理特徵和標籤
    for i, unique_id in enumerate(unique_id_list):
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = feature_file.iloc[(i * 27):(i * 27) + 27, feature_importance[model_name]]
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)
    
    # 檢查訓練集是否為空
    if x_train.empty:
        raise ValueError("x_train is empty. Check unique_id matching or player_id splitting.")
    
    # 特徵縮放
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    # 定義每組數據的大小
    group_size = 27

    def RF_binary(X_train, y_train, X_test, y_test, category, n, max_depth):
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        
        save_dir = 'Models/Exp_3_2'
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, f'{save_dir}/{category}.pkl')
        
        predicted = clf.predict_proba(X_test)
        predicted = [predicted[i][0] for i in range(len(predicted))]
        num_groups = len(predicted) // group_size
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        y_pred = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f'{category} AUC: {auc_score}')
        
        y_pred_labels = [1 if p > 0.4 else 0 for p in y_pred]  # 降低閾值
        f1 = f1_score(y_test_agg, y_pred_labels, average='macro')
        print(f'{category} Macro F1 Score: {f1}')
        print(f'{category} Confusion Matrix:\n{confusion_matrix(y_test_agg, y_pred_labels)}')

    def RF_multiary(X_train, y_train, X_test, y_test, category, n, max_depth):
        # 動態計算 SMOTE 採樣策略
        if category == "years":
            class_counts = pd.Series(y_train).value_counts()
            sampling_strategy = {
                0: max(12000, class_counts.get(0, 0)),  # 球齡 0 年
                1: max(19251, class_counts.get(1, 0)),  # 球齡 1 年（避免過度放大）
                2: max(17000, class_counts.get(2, 0))   # 球齡 2 年
            }
        else:
            sampling_strategy = 'auto'
        
        smote = SMOTE(
            random_state=42, 
            k_neighbors=5, 
            sampling_strategy=sampling_strategy
        )
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        clf = BalancedRandomForestClassifier(
            n_estimators=n,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=42,
            sampling_strategy='auto',
            replacement=True
        )
        clf.fit(X_train_resampled, y_train_resampled)
        
        save_dir = 'Models/Exp_4_2'
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, f'{save_dir}/{category}.pkl')
        
        feature_importances = pd.Series(clf.feature_importances_, index=x_train.columns)
        print(f'{category} Feature Importances:\n{feature_importances.sort_values(ascending=False)}')
        
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print(f'{category} Multiary AUC: {auc_score}')
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test_agg, y_pred_labels, average='macro')
        print(f'{category} Macro F1 Score: {f1}')
        print(f'{category} Confusion Matrix:\n{confusion_matrix(y_test_agg, y_pred_labels)}')

    def CB_binary(X_train, y_train, X_test, y_test, category, iterations, depth):
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        clf = CatBoostClassifier(
            task_type='GPU', 
            verbose=0, 
            random_state=42,
            iterations=iterations, 
            depth=depth, 
            class_weights=weights,
            l2_leaf_reg=1  # 進一步降低正則化
        )
        clf.fit(X_train, y_train)
        
        save_dir = 'Models/Exp_4_2'
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, f'{save_dir}/{category}.pkl')
        
        predicted = clf.predict_proba(X_test)
        predicted = [predicted[i][0] for i in range(len(predicted))]
        num_groups = len(predicted) // group_size
        if sum(predicted[:group_size]) / group_size > 0.5:
            y_pred = [max(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        else:
            y_pred = [min(predicted[i*group_size: (i+1)*group_size]) for i in range(num_groups)]
        y_pred = [1 - x for x in y_pred]
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        print(f'{category} AUC: {auc_score}')
        
        y_pred_labels = [1 if p > 0.4 else 0 for p in y_pred]  # 降低閾值
        f1 = f1_score(y_test_agg, y_pred_labels, average='macro')
        print(f'{category} Macro F1 Score: {f1}')
        print(f'{category} Confusion Matrix:\n{confusion_matrix(y_test_agg, y_pred_labels)}')

    def CB_multiary(X_train, y_train, X_test, y_test, category, iterations, depth):
        # 動態計算 SMOTE 採樣策略
        if category == "level":
            class_counts = pd.Series(y_train).value_counts()
            sampling_strategy = {
                2: max(10000, class_counts.get(2, 0)),  # 等級 3
                3: max(6000, class_counts.get(3, 0))    # 等級 4
            }
            smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        clf = CatBoostClassifier(
            task_type='GPU', 
            verbose=0, 
            random_state=42,
            iterations=iterations, 
            depth=depth, 
            class_weights=weights,
            l2_leaf_reg=3
        )
        clf.fit(X_train, y_train)
        
        save_dir = 'Models/Exp_4_2'
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(clf, f'{save_dir}/{category}.pkl')
        
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = []
        
        for i in range(num_groups):
            group_pred = predicted[i*group_size: (i+1)*group_size]
            num_classes = len(np.unique(y_train))
            class_sums = [sum([group_pred[k][j] for k in range(group_size)]) for j in range(num_classes)]
            chosen_class = np.argmax(class_sums)
            candidate_probs = [group_pred[k][chosen_class] for k in range(group_size)]
            best_instance = np.argmax(candidate_probs)
            y_pred.append(group_pred[best_instance])
        
        y_test_agg = [y_test[i*group_size] for i in range(num_groups)]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        print(f'{category} Multiary AUC: {auc_score}')
        
        y_pred_labels = np.argmax(y_pred, axis=1)
        f1 = f1_score(y_test_agg, y_pred_labels, average='macro')
        print(f'{category} Macro F1 Score: {f1}')
        print(f'{category} Confusion Matrix:\n{confusion_matrix(y_test_agg, y_pred_labels)}')

    le = LabelEncoder()
    if model_name == "gender":
        y_train_le_gender = le.fit_transform(y_train['gender'])
        y_test_le_gender = le.transform(y_test['gender'])
        CB_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender, 'gender', 2000, 2)
    elif model_name == "hold":
        y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
        y_test_le_hold = le.transform(y_test['hold racket handed'])
        CB_binary(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold, 'hold', 1000, 6)
    elif model_name == "year":
        y_train_le_years = le.fit_transform(y_train['play years'])
        y_test_le_years = le.transform(y_test['play years'])
        RF_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years, "years", 700, 10)  # 調整參數
    elif model_name == "level":
        y_train_le_level = le.fit_transform(y_train['level'])
        y_test_le_level = le.transform(y_test['level'])
        CB_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level, "level", 2500, 4)  # 調整參數

    save_dir = 'Models/Exp_4_2'
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, f'{save_dir}/{model_name}_scaler.pkl')

if __name__ == '__main__':
    categories = ["gender", "hold", "year", "level"]
    for category in categories:
        main(category)