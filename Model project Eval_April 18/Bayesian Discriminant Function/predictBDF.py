import numpy as np
from BDF import BayesianDiscriminantFunction

# Create a 2-class dataset
X = np.concatenate([np.random.normal(loc=[1, 1], scale=0.5, size=(50, 2)), 
                    np.random.normal(loc=[-1, -1], scale=0.5, size=(50, 2))])
y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)

# Create an instance of BayesianDiscriminantFunction and fit the data
model = BayesianDiscriminantFunction(num_classes=2)
model.fit(X, y)

# Generate some test data and predict the labels
X_test = np.random.normal(loc=[0, 0], scale=0.5, size=(20, 2))
y_pred = model.predict(X_test)
print(y_pred)
