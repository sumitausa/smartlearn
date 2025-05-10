"""
User Performance Classification Example Workflow
------------------------------------------------
This script demonstrates:
1. Generating synthetic user performance data
2. Assigning learning level labels
3. Training a Random Forest classifier
4. Evaluating the model
5. Predicting new user levels
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import random
import joblib

# 1. Generate synthetic user performance data
def generate_synthetic_data(num_users=300, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    data = []

    # Define user archetypes with more realistic characteristics and overlap
    archetypes = {
        'Advanced': {
            'score_mean': 85, 'score_std': 12,  # More variance
            'quiz_mean': 12, 'quiz_std': 4,    # More spread
            'hint_mean': 0.2, 'hint_std': 0.15, # More overlap
            'retry_mean': 0.8, 'retry_std': 0.6 # More variance
        },
        'Intermediate': {
            'score_mean': 72, 'score_std': 13,
            'quiz_mean': 8, 'quiz_std': 4,
            'hint_mean': 0.4, 'hint_std': 0.2,
            'retry_mean': 1.5, 'retry_std': 0.9
        },
        'Beginner': {
            'score_mean': 58, 'score_std': 15,
            'quiz_mean': 5, 'quiz_std': 3,
            'hint_mean': 0.65, 'hint_std': 0.25,
            'retry_mean': 2.2, 'retry_std': 1.2
        }
    }
    
    # Generate synthetic users with more natural variation
    for _ in range(num_users):
        # Weighted random selection favoring intermediate
        weights = [0.25, 0.5, 0.25]  # Advanced, Intermediate, Beginner
        archetype = random.choices(list(archetypes.keys()), weights=weights)[0]
        arch = archetypes[archetype]
        
        # Generate characteristics with natural variation
        avg_score = max(0, min(100, np.random.normal(
            arch['score_mean'], arch['score_std']
        )))
        
        num_quizzes = max(1, int(np.random.normal(
            arch['quiz_mean'], arch['quiz_std']
        )))
        
        hint_usage = max(0, min(1, np.random.normal(
            arch['hint_mean'], arch['hint_std']
        )))
        
        retries = max(0, np.random.normal(
            arch['retry_mean'], arch['retry_std']
        ))
        
        # More nuanced level determination with ambiguous cases
        if avg_score >= 85 and hint_usage < 0.3 and num_quizzes > 10:
            level = 'Advanced'
        elif 78 <= avg_score < 85 and hint_usage < 0.35:
            # Borderline Advanced/Intermediate
            level = random.choices(['Advanced', 'Intermediate'], weights=[0.4, 0.6])[0]
        elif avg_score >= 65 and hint_usage < 0.55:
            level = 'Intermediate'
        elif 60 <= avg_score < 65:
            # Borderline Intermediate/Beginner
            level = random.choices(['Intermediate', 'Beginner'], weights=[0.4, 0.6])[0]
        else:
            level = 'Beginner'
            
        data.append([avg_score, num_quizzes, hint_usage, retries, level])
    
    columns = ['avg_score', 'num_quizzes', 'hint_usage', 'retries', 'level']
    return pd.DataFrame(data, columns=columns)

# 2. Prepare data for training
def prepare_data(df):
    X = df.drop('level', axis=1)  # Features: avg_score, num_quizzes, hint_usage, retries
    y = df['level']               # Target: learning level
    return X, y

# 3. Train and evaluate multiple classifiers
def compare_classifiers(X_train, X_test, y_train, y_test):
    # Scale features for better performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers with more robust parameters
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance'
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
    }
    
    best_score = 0
    best_clf = None
    best_scaler = None
    
    print("Comparing Classifiers:")
    print("-" * 50)
    
    for name, clf in classifiers.items():
        # K-fold cross validation
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
        
        # Train on full training set
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"Test Accuracy: {accuracy:.3f}")
        
        # Cross-validation scores
        print(f"Cross-validation scores by fold:")
        for fold, score in enumerate(cv_scores, 1):
            print(f"Fold {fold}: {score:.3f}")
        print(f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"Min CV Score: {cv_scores.min():.3f}")
        print(f"Max CV Score: {cv_scores.max():.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        
        print("\nConfusion Matrix:")
        print("Labels: [Advanced, Beginner, Intermediate]")
        
        # Calculate and print AUC scores for each class
        y_test_bin = label_binarize(y_test, classes=['Advanced', 'Beginner', 'Intermediate'])
        
        print("\nAUC Scores:")
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test_scaled)
            for i, class_name in enumerate(['Advanced', 'Beginner', 'Intermediate']):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                auc_score = auc(fpr, tpr)
                print(f"{class_name}: {auc_score:.3f}")
        
            # Calculate macro-average AUC
            macro_auc = np.mean([auc(roc_curve(y_test_bin[:, i], y_score[:, i])[0],
                                    roc_curve(y_test_bin[:, i], y_score[:, i])[1])
                                for i in range(3)])
            print(f"Macro-average AUC: {macro_auc:.3f}")
        
        # Print classification report
        report = classification_report(y_test, y_pred)
        print("\nClassification Report:")
        print(report)
        
        # Track best classifier based on cross-validation mean
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_clf = clf
            best_scaler = scaler
    
    print("\nBest Classifier:")
    print(f"Model: {type(best_clf).__name__}")
    print(f"Cross-validation Score: {best_score:.3f}")
    
    # Save the Gradient Boosting model specifically
    gb_model = classifiers['Gradient Boosting']
    gb_model.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    joblib.dump(gb_model, 'gradient_boosting_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    print("\nGradient Boosting model has been saved as 'gradient_boosting_model.joblib'")
    print("Scaler has been saved as 'scaler.joblib'")
    
    return gb_model, scaler

def plot_learning_curves(classifiers, X, y):
    # Calculate number of rows needed (2 classifiers per row)
    n_classifiers = len(classifiers)
    n_rows = (n_classifiers + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    if n_rows > 1:
        axes = axes.ravel()
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx] if n_rows > 1 else axes[idx % 2]
        
        train_sizes, train_scores, test_scores = learning_curve(
            clf, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
        ax.fill_between(train_sizes, train_mean - train_std,
                       train_mean + train_std, alpha=0.1, color='r')
        ax.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
        ax.fill_between(train_sizes, test_mean - test_std,
                       test_mean + test_std, alpha=0.1, color='g')
        
        ax.set_title(f'{name} Learning Curve')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend(loc='lower right')
        ax.set_ylim([0.4, 1.1])  # Set y-axis limits for better visualization
    
    # If odd number of classifiers, remove the last empty subplot
    if n_classifiers % 2 == 1 and n_rows > 1:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('model_learning_curves.png')
    plt.close()

def plot_confusion_matrices(classifiers, X_test, y_test):
    # Calculate number of rows needed (2 classifiers per row)
    n_classifiers = len(classifiers)
    n_rows = (n_classifiers + 1) // 2  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))
    if n_rows > 1:
        axes = axes.ravel()
        
    class_names = ['Advanced', 'Beginner', 'Intermediate']
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx]
        
        # Get predictions
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        
        # Add title
        ax.set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.1%}', pad=20)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add numbers and percentages in cells
        thresh = cm.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                ax.text(j, i, text,
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # Add axis labels
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Remove empty subplots
    for idx in range(n_classifiers, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(classifiers, X_test, y_test):
    # Create a 2x2 subplot layout
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(2, 2, height_ratios=[1, 1])
    
    # First plot: Educational example with annotations
    ax0 = fig.add_subplot(gs[0, :])
    
    # Get ROC curve for the best classifier (Gradient Boosting) for Advanced level
    best_clf = classifiers['Gradient Boosting']
    y_bin = label_binarize(y_test, classes=['Advanced', 'Beginner', 'Intermediate'])
    y_score = best_clf.predict_proba(X_test)
    fpr, tpr, _ = roc_curve(y_bin[:, 0], y_score[:, 0])
    roc_auc = auc(fpr, tpr)
    
    # Plot main ROC curve
    ax0.plot(fpr, tpr, 'b-', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax0.plot([0, 1], [0, 1], 'r--', lw=2, label='Random Chance')
    
    # Add educational annotations
    ax0.annotate('Perfect Classification\n(Best Possible)', xy=(0, 1), xytext=(0.2, 1),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax0.annotate('Random Chance Line\n(AUC = 0.5)', xy=(0.5, 0.5), xytext=(0.6, 0.4),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    ax0.annotate('Better Performance\n(Higher AUC)', xy=(0.2, 0.8), xytext=(0.4, 0.7),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    
    # Add explanation text boxes
    ax0.text(0.05, 0.4, 'True Positive Rate (TPR):\nProportion of actual positives\ncorrectly identified',
             bbox=dict(facecolor='white', alpha=0.8))
    ax0.text(0.7, 0.05, 'False Positive Rate (FPR):\nProportion of negatives\nincorrectly identified',
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax0.set_xlim([-0.05, 1.05])
    ax0.set_ylim([-0.05, 1.05])
    ax0.set_xlabel('False Positive Rate')
    ax0.set_ylabel('True Positive Rate')
    ax0.set_title('How to Read an ROC Curve - Educational Example')
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc='lower right')
    
    # Bottom plots: Actual model comparisons for each class
    class_names = ['Advanced', 'Beginner', 'Intermediate']
    axes = [fig.add_subplot(gs[1, i]) for i in range(2)]
    axes.append(fig.add_subplot(gs[1, -1]))
    
    classifier_colors = {
        'Gradient Boosting': 'red',
        'Random Forest': 'blue',
        'SVM': 'green',
        'KNN': 'purple',
        'Logistic Regression': 'orange'
    }
    
    # Plot ROC curves for each class
    for name, clf in classifiers.items():
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(X_test)
        else:
            y_score = clf.decision_function(X_test)
            if y_score.ndim == 1:
                y_score = np.vstack([-y_score, y_score]).T
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            axes[i].plot(fpr, tpr, color=classifier_colors[name], lw=2,
                        label=f'{name} (AUC = {roc_auc:.2f})')
            axes[i].plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Customize each subplot
    for i, ax in enumerate(axes):
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {class_names[i]}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', prop={'size': 8})
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(classifiers, feature_names, X, y):
    n_features = len(feature_names)
    n_classifiers = len(classifiers)
    
    # Create a figure with a subplot for each classifier
    fig, axes = plt.subplots(n_classifiers, 1, figsize=(12, 5*n_classifiers))
    if n_classifiers == 1:
        axes = [axes]
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        ax = axes[idx]
        
        # Get feature importance
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            # For linear models like Logistic Regression
            importances = np.abs(clf.coef_).mean(axis=0)
        else:
            # For models without direct feature importance (like KNN)
            importances = np.zeros(n_features)
            for i in range(n_features):
                X_temp = X.copy()
                X_temp[:, i] = np.random.permutation(X_temp[:, i])
                score_orig = clf.score(X, y)
                score_perm = clf.score(X_temp, y)
                importances[i] = score_orig - score_perm
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot
        bars = ax.bar(range(n_features), importances[indices])
        ax.set_title(f'{name} Feature Importance')
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45)
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('feature_importance_all.png', bbox_inches='tight')
    plt.close()

# 5. Test predictions on example users
def predict_new_users(clf):
    # Example users with varying performance patterns
    new_users = pd.DataFrame({
        'avg_score': [90, 75, 50],      # High, medium, low scores
        'num_quizzes': [15, 8, 5],      # High, medium, low engagement
        'hint_usage': [0.1, 0.3, 0.8],  # Low, medium, high hint usage
        'retries': [1, 2, 4]            # Low, medium, high retries
    })
    
    preds = clf.predict(new_users)
    print("\nPredictions for new users:")
    for i, pred in enumerate(preds):
        print(f"User {i+1}: {pred}")

# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    data = generate_synthetic_data()
    
    # Prepare features and target
    X = data.drop('level', axis=1)
    y = data['level']
    feature_names = X.columns
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, random_state=42, probability=True
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance'
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, class_weight='balanced', max_iter=1000, random_state=42
        )
    }
    
    # Train all classifiers
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
    
    # Compare classifiers and get the best one
    best_clf, best_scaler = compare_classifiers(X_train, X_test, y_train, y_test)
    
    print("\nGenerating visualization plots...")
    
    # Generate plots
    plot_learning_curves(classifiers, X_train_scaled, y_train)
    plot_confusion_matrices(classifiers, X_test_scaled, y_test)
    plot_roc_curves(classifiers, X_test_scaled, y_test)
    plot_feature_importance(classifiers, feature_names, X_test_scaled, y_test)
    
    print("Plots have been saved:")
    print("1. model_learning_curves.png - Shows learning curves for all models")
    print("2. confusion_matrices.png - Shows confusion matrices for all models")
    print("3. roc_curves.png - Shows ROC curves for all models")
    print("4. feature_importance_all.png - Shows feature importance for all models")
    
    print("\nTesting Best Classifier:")
    # Example new users
    new_users = pd.DataFrame([
        [65, 3, 0.8, 2.5],  # Likely Beginner
        [78, 7, 0.3, 1.2],  # Likely Intermediate
        [55, 2, 0.9, 3.0]   # Likely Beginner
    ], columns=['avg_score', 'num_quizzes', 'hint_usage', 'retries'])
    
    # Scale the new data using the same scaler
    new_users_scaled = best_scaler.transform(new_users)
    
    # Make predictions
    predictions = best_clf.predict(new_users_scaled)
    
    print("\nPredictions for new users:")
    for i, pred in enumerate(predictions, 1):
        print(f"User {i}: {pred}")

    # Save model and scaler for Flask app
    joblib.dump(best_clf, "user_level_classifier.joblib")
    joblib.dump(best_scaler, "user_level_scaler.joblib")

# Utility for Flask app
def load_classifier(model_path="user_level_classifier.joblib"):
    import joblib
    return joblib.load(model_path)
