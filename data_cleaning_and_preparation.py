import pandas as pd
import numpy as np

# Load data
df = pd.read_excel('Crash Data AA.xlsx')
df.columns = [c.replace(' ', '_') for c in df.columns]

# Define the cleanup mapping
target_mapping = {
    'Fatal': 'Fatal',
    'Minor': 'Minor',
    'Serious': 'Serious',
    'P.D.O': 'Property_Damage',
    'PDO': 'Property_Damage',
    'POD': 'Property_Damage',
    'property damage only': 'Property_Damage',
    'Serious and Empty Ones': 'Serious' # Assuming these were serious
}

# Apply mapping
df['Accident_Type'] = df['Accident_Type'].str.strip() # Remove hidden spaces
df['Accident_Type'] = df['Accident_Type'].map(target_mapping)

# Drop rows where Accident_Type is still NaN (the truly empty ones)
df = df.dropna(subset=['Accident_Type'])

print("Cleaned Target Classes:\n", df['Accident_Type'].value_counts())

# Calculate percentage of missing values per column
missing_pct = df.isnull().mean() * 100

# Identify columns with more than 60% missing data
cols_to_drop = missing_pct[missing_pct > 60].index.tolist()

print(f"Dropping columns with >50% missing values: {cols_to_drop}")
df = df.drop(columns=cols_to_drop)

# Also drop unique IDs or useless text like 'Date'
df = df.drop(columns=['Accident_ID', 'Date'], errors='ignore')

# Fill Categorical columns with 'Unknown'
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill Numerical columns with Median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print("Missing values after cleaning:", df.isnull().sum().sum())

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(x='Accident_Type', data=df)
plt.title('Cleaned Accident Type Distribution')
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)

# Only plot if there are actually numeric columns to show
if not correlation_matrix.empty:
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()
else:
    print("No numeric columns found to correlate. You'll need to encode data first!")

from sklearn.preprocessing import LabelEncoder

# 1. Define Target and Features
y = df['Accident_Type']
X = df.drop(columns=['Accident_Type'])

# 2. CRITICAL: Explicitly drop the leakage columns
# We define them and verify they are gone
leakage_cols = [
    'Number_of_fatalities',
    'Number_of_sever_injuries',
    'Number_of_minor_injuries',
    'Number_of_Casualties'
]

# Drop them and print confirmation
X = X.drop(columns=leakage_cols, errors='ignore')
print(f"Leakage columns removed. 'Number_of_fatalities' in X: {'Number_of_fatalities' in X.columns}")

# 3. Label Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. One-Hot Encode Features
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)

print(f"Shape of features after One-Hot Encoding: {X_encoded.shape}")

from sklearn.model_selection import train_test_split

# Split: 80% Train, 20% Test
# We use stratify=y_encoded to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Optional: If you need a Validation set (e.g., for Neural Networks),
# you split the Training set again (e.g., getting 10% of total original data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.125, # 0.125 of 80% = 10% of total
    random_state=42,
    stratify=y_train
)

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Testing shape:  {X_test.shape}")

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize the Scaler
# Use StandardScaler if you assume normal distribution (common for tabular data)
# Use MinMaxScaler if you need strict 0-1 bounds (e.g., for image pixel data or Neural Nets)
scaler = StandardScaler()

# Fit on Train, Transform on Train
X_train_scaled = scaler.fit_transform(X_train)

# Transform only on Val and Test (using the Train mean/std)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability (optional, but helpful for debugging)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

print("Data Preprocessing Complete.")
print("First 5 rows of scaled training data:")
print(X_train_scaled.head())

from imblearn.over_sampling import SMOTE

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
# Change the variable names here to match what the NN expects
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"Resampled Training Shape: {X_train_resampled.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Get input shape
input_dim = X_train_resampled.shape[1]
num_classes = len(le.classes_)

# 1. Define the Neural Network Architecture
model = Sequential([
    # Input Layer & Hidden Layer 1
    # 64 neurons, ReLU activation for non-linearity
    Dense(64, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(), # Stabilizes learning
    Dropout(0.3), # Prevents overfitting by randomly turning off neurons

    # Hidden Layer 2
    Dense(32, activation='relu'),

    # Hidden Layer 3
    Dense(16, activation='relu'),

    # Output Layer
    # Softmax is mandatory for multi-class classification
    Dense(num_classes, activation='softmax')
])

model.summary()

# Import optimizer
from tensorflow.keras.optimizers import Adam

# 2. Compile the Model with a smaller learning rate
# A lower rate (0.001 or 0.0001) helps prevent the loss from exploding
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Setup Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 4. Train
history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plot Training History
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.show()

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Get Predictions
y_pred_probs = model.predict(X_test_scaled)

# 2. Convert Probabilities to Class Labels
y_pred = np.argmax(y_pred_probs, axis=1)

# 3. Print Report
target_names = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# 4. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Neural Network Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()