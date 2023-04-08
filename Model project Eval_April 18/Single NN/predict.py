import numpy as np
from SingleLayerNN import SingleLayerNN

# Create a 2-class dataset
X = np.concatenate([np.random.normal(loc=[1, 1], scale=0.5, size=(50, 2)), 
                    np.random.normal(loc=[-1, -1], scale=0.5, size=(50, 2))])
y = np.concatenate([np.zeros(50), np.ones(50)]).reshape(-1, 1)

# Create an instance of SingleLayerNN and fit the data
model = SingleLayerNN(input_dim=2, output_dim=1, learning_rate=0.1)
model.fit(X, y, num_epochs=1000)

# Generate some test data and predict the outputs
X_test = np.random.normal(loc=[0, 0], scale=0.5, size=(20, 2))
y_pred = model.predict(X_test)
print(y_pred)
