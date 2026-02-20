import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

def walk_forward_ensemble(X, y, train_window=1260, step=63):
    """
    Strict Expanding Window Walk-Forward Validation.
    Retrains models periodically using only historical data.
    """
    predictions = []
    probabilities = []
    
    # Define Base Models
    clf1 = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=15, random_state=42)
    clf3 = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    
    # Voting Ensemble (Soft voting for probability calibration)
    ensemble = VotingClassifier(estimators=[
        ('lr', clf1), ('rf', clf2), ('xgb', clf3)
    ], voting='soft')
    
    scaler = StandardScaler()

    # Walk-Forward Loop
    for i in range(train_window, len(X), step):
        # Training Set: [0 to i]
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        
        # Test Set (Out of Sample): [i to i + step]
        test_end = min(i + step, len(X))
        X_test = X.iloc[i:test_end]
        
        # Scale strictly on training data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Ensemble
        ensemble.fit(X_train_scaled, y_train)
        
        # Predict
        probs = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        # Store results mapped to actual dates
        idx = X_test.index
        probabilities.extend(pd.Series(probs, index=idx))
    
    # Return as DataFrame
    res = pd.DataFrame({'Prob_Risk_On': probabilities}, index=X.index[train_window:])
    return res, ensemble, scaler
