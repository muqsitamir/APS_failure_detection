# Model Comparison Report

Best overall model: **XGBoost** with mean CV balanced accuracy 0.9608

```
             model  mean_test_score  std_test_score  mean_train_score  std_train_score                                                                                                                           params
LogisticRegression         0.945062        0.007389          0.955595         0.000797                                                                                           {'clf__C': 0.01, 'clf__penalty': 'l1'}
      RandomForest         0.944208        0.008328          0.985161         0.000694                       {'clf__max_depth': 10, 'clf__max_features': 'sqrt', 'clf__min_samples_leaf': 10, 'clf__n_estimators': 600}
           XGBoost         0.960804        0.008529          0.987622         0.000756 {'clf__colsample_bytree': 0.8, 'clf__learning_rate': 0.05, 'clf__max_depth': 3, 'clf__n_estimators': 300, 'clf__subsample': 1.0}
```