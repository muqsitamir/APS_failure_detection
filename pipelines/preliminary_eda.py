from utils.data_analysis_utils import inspect_data, check_class_distribution, check_missing_values
from utils.data_utils import load_data_for_eda

if __name__ == "__main__":
    X_train, y_train, X_test = load_data_for_eda()
    
    inspect_data(X_train, "X_train")
    
    check_class_distribution(y_train, "y_train")
    
    check_missing_values(X_train, "X_train")