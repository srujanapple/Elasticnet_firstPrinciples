import csv
import numpy as np
import matplotlib.pyplot as plt
from ElasticNet import ElasticNetModel

def test_predict():
    
    model = ElasticNetModel()
    
    # Load data
    data = []
    with open("small_test.csv", "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    
    # features and target
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(v) for datum in data for k, v in datum.items() if k == 'y'])
    
    results = model.fit(X, y)
    preds = results.predict(X)
    
    assert preds is not None, "Predictions should not be None"
    assert len(preds) == len(X), "Number of predictions should match number of samples"
    assert isinstance(preds, np.ndarray), "Predictions should be numpy array"
    
    
    print("\nModel Details:")
    print(f"Model coefficients: {results.coef_}")
    print(f"Model intercept: {results.intercept_}")
    print(f"R²: {results.r_squared_:.4f}")
    print(f"MSE: {results.mse_:.4f}")
    
    #plots
    try:
        #Convergence
        print("\nCreating convergence plot...")
        results.plot_convergence()
        
        #Predictions vs Actual
        print("Creating predictions plot...")
        results.plot_predictions(X, y)
        
        #Feature Importance
        print("Creating feature importance plot...")
        results.plot_feature_importance(['x_0', 'x_1', 'x_2'])
        
        print("All plots created successfully!")
    except Exception as e:
        print(f"Error creating plots: {str(e)}")

if __name__ == "__main__":
    print("Starting ElasticNet Model Test")
    test_predict()
    print("\nTest completed!")
    plt.show()  
