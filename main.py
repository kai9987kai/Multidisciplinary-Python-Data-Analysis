import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

# Numerical computation and data manipulation
# Generate a synthetic dataset
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.rand(100, 1)

# Organize the data
df = pd.DataFrame(np.hstack([X, y]), columns=['X', 'y'])

# Machine learning
# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Train a simple linear regression model
model = LinearRegression()
model.fit(train_df[['X']], train_df['y'])

# Visualization
# Plot the data and the model's predictions
plt.scatter(train_df['X'], train_df['y'], color='blue', label='Actual')
plt.plot(train_df['X'], model.predict(train_df[['X']]), color='red', label='Predicted')
plt.legend()
plt.show()

# Natural language processing
# Tokenize and count words in a sentence
sentence = "This is a simple sentence for tokenization."
tokens = word_tokenize(sentence)
(tokens, len(tokens))
