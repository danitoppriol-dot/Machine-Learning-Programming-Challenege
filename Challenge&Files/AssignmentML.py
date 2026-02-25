import numpy as np
import csv
from collections import Counter
import random
import math

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 

class CustomDecisionTree:

    def __init__(self, max_depth=10, min_samples_split=10, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features if max_features is not None else 5
        self.root = None
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]
        
        if self.max_features > self.n_features or self.max_features is None:
            self.max_features = int(np.sqrt(self.n_features))
        if self.max_features == 0:
            self.max_features = 1
        
        self.root = self._grow_tree(X, y, 0)
        return self

    def _calculate_leaf_value(self, y):
        counts = Counter(y)
        return counts.most_common(1)[0][0]

    def _get_random_features(self):
        feature_indices = list(range(self.n_features))
        return random.sample(feature_indices, min(len(feature_indices), self.max_features))

    def _find_best_split(self, X, y):
        best_gini = 1.0 
        best_feature, best_threshold = None, None
        
        random_features = self._get_random_features()

        for feature_idx in random_features:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                left_mask = X_column <= threshold
                y_left = y[left_mask]
                y_right = y[~left_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                n_left, n_right = len(y_left), len(y_right)
                n_total = n_left + n_right
                
                gini_left = 1.0 - sum((np.sum(y_left == c) / n_left)**2 for c in self.classes_)
                gini_right = 1.0 - sum((np.sum(y_right == c) / n_right)**2 for c in self.classes_)

                weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini 
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=self._calculate_leaf_value(y))

        feature_idx, threshold = self._find_best_split(X, y)

        if feature_idx is None:
            return Node(value=self._calculate_leaf_value(y))

        left_mask = X[:, feature_idx] <= threshold
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]

        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return Node(feature_idx, threshold, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])


class CustomRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=10, max_features=None, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.seed = seed
        self.trees = []
        self.classes_ = None

    def fit(self, X, y):
        np.random.seed(self.seed)
        N = X.shape[0]
        self.classes_ = np.unique(y)
        self.trees = []

        for i in range(self.n_estimators):
            
            bootstrap_indices = np.random.choice(N, size=N, replace=True)
            X_sample, y_sample = X[bootstrap_indices], y[bootstrap_indices]
            
            tree = CustomDecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features, 
                min_samples_split=10
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        return self

    def predict(self, X):

        predictions = np.array([tree.predict(X) for tree in self.trees]).T 
        
        y_final = np.array([Counter(sample_preds).most_common(1)[0][0] 
                             for sample_preds in predictions])
        return y_final

def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))
    return xTr,yTr,xTe,yTe,trIdx,teIdx

def read_csv_manual(filename, has_target):
    data = []
    titol = []
    try:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            titol = next(reader) 
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato: {filename}.")
        exit()
            
    if has_target:
        X = [row[1:] for row in data]
        y = [row[0] for row in data]
    else:
        X = data
        y = None
        
    return X, y

def preprocess_data_manual(X_raw, y_raw, is_training, train_mapper=None):
    if is_training:
        unique_labels = sorted(list(set(y_raw)))
        label_encoder = {label: i for i, label in enumerate(unique_labels)}
        numeric_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12]
        x7_values = [row[6] for row in X_raw if row[6] != '']
        mode_x7 = Counter(x7_values).most_common(1)[0][0]
        all_x7_values = set(x7_values)
        category_mapper = {cat: i for i, cat in enumerate(all_x7_values)}
        medians = {}
        for i in numeric_indices:
            vals = [float(row[i]) for row in X_raw if row[i] != '']
            medians[i] = np.median(vals)
    else:
        label_encoder = train_mapper['label_encoder']
        mode_x7 = train_mapper['mode_x7']
        medians = train_mapper['medians']
        category_mapper = train_mapper['category_mapper']
        
    processed_X = []
    for row in X_raw:
        new_row = []
        for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12]:
            val = row[i]
            new_row.append(medians[i] if val == '' else float(val))
        cat_val = row[6] if row[6] != '' else mode_x7
        ohe_vector = [0.0] * len(category_mapper)
        if cat_val in category_mapper:
            ohe_vector[category_mapper[cat_val]] = 1.0
        new_row.extend(ohe_vector)
        bool_val = row[11]
        if bool_val == 'True':
            new_row.append(1.0)
        elif bool_val == 'False':
            new_row.append(0.0)
        else:
            new_row.append(1.0) 
        processed_X.append(new_row)
    processed_X = np.array(processed_X, dtype=float)

    if is_training:
        X_mean = np.mean(processed_X, axis=0)
        X_std = np.std(processed_X, axis=0)
        X_std[X_std == 0] = 1 
        output_X = (processed_X - X_mean) / X_std
        output_Y = np.array([label_encoder[y] for y in y_raw], dtype=int)
        output_mapper = {
            'label_encoder': label_encoder,
            'inverse_label_encoder': {v: k for k, v in label_encoder.items()},
            'medians': medians,
            'mode_x7': mode_x7,
            'category_mapper': category_mapper,
            'X_mean': X_mean,
            'X_std': X_std
        }
    else:
        X_mean = train_mapper['X_mean']
        X_std = train_mapper['X_std']
        output_X = (processed_X - X_mean) / X_std
        output_Y = None
        output_mapper = None

    return output_X, output_Y, output_mapper

def cross_val_score_custom(model_class, X, y, folds=5, **kwargs):
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    fold_sizes = np.full(folds, len(y) // folds, dtype=int)
    fold_sizes[:len(y) % folds] += 1
    current = 0
    scores = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        current = stop
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        scores.append(np.mean(y_pred == y_val))
    return np.mean(scores)

def solve():
    X_train_raw, y_train_raw = read_csv_manual("TrainOnMe_orig.csv", has_target=True)
    X_eval_raw, _ = read_csv_manual("EvaluateOnMe.csv", has_target=False)
    all_data, y_full_encoded, train_mapper = preprocess_data_manual(
        X_train_raw, y_train_raw, is_training=True
    )
    
    n_estimators = 200
    max_depth = 5  
    max_features = 5 

    rf_classifier = CustomRandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        max_features=max_features,
        seed=42
    )

    print("let's goo")
    cv_accuracy = cross_val_score_custom(
        CustomRandomForestClassifier,
        all_data,
        y_full_encoded,
        folds=5,
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        seed=42
    )
    xTr, yTr, xTe, yTe, _, _ = trteSplitEven(all_data, y_full_encoded, 0.8, seed=42)


    val_model = rf_classifier.fit(xTr, yTr) 
    y_pred = val_model.predict(xTe) 
    
    test_accuracy = np.mean((y_pred == yTe).astype(float))

    classifier = CustomRandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        max_features=max_features,
        seed=42
    )
    final_model = classifier.fit(all_data, y_full_encoded) 
    
    y_pred_training = final_model.predict(all_data)
    training_accuracy = np.mean((y_pred_training == y_full_encoded).astype(float))
    

    X_eval_processed, _, _ = preprocess_data_manual(
        X_eval_raw, None, is_training=False, train_mapper=train_mapper
    )

    y_pred_encoded = final_model.predict(X_eval_processed)
    inverse_le = train_mapper['inverse_label_encoder']
    y_pred_final = [inverse_le[i] for i in y_pred_encoded]

    predicted_labels_content = "\n".join(y_pred_final)
    output_filename = "predicted_labels.txt"
    
    with open(output_filename, "w") as f:
        f.write(predicted_labels_content)

    print(f"accuracy estimated (on 20% test): {test_accuracy * 100:.2f}%")
    print(f"accuracy estimated (on 80% training): {training_accuracy * 100:.2f}%")
    print(f"accuracy avg estimated (5-fold cross val): {cv_accuracy * 100:.2f}%")
    print('parameters:')
    print(n_estimators,max_depth,max_features)
    
    
if __name__ == "__main__":
    solve()
