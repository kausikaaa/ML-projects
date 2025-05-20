# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split

# Example data (you can replace this with your own dataset)
data = {
    'text': [
        "I love pizza",
        "Win a free iPhone now",
        "Let's schedule the meeting",
        "You won a lottery prize",
        "Project deadline is tomorrow",
        "Congratulations, claim your prize",
        "Work hard, achieve more"
    ],
    'label': [0, 1, 0, 1, 0, 1, 0]  # 0 = not spam, 1 = spam
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split into features (X) and labels (y)
X = df['text']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Show the training data
print("Training Texts:\n", X_train)
print("\nTraining Labels:\n", y_train)