"""
Data Cleaning and Neural Network Training for Crash Analysis
Preprocesses crash data, handles class imbalance, and trains a neural network classifier.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load and clean column names
df = pd.read_excel('Crash Data AA.xlsx')
df.columns = [c.replace(' ', '_') for c in df.columns]

# Standardize accident types
target_mapping = {
    'Fatal': 'Fatal',
    'Minor': 'Minor',
    'Serious': 'Serious',
    'P.D.O': 'Property_Damage',
    'PDO': 'Property_Damage',
    'POD': 'Property_Damage',
    'property damage only': 'Property_Damage',
    'Serious and Empty Ones': 'Serious'
}

df['Accident_Type'] = df['Accident_Type'].str.strip().map(target_mapping)
df = df.dropna(subset=['Accident_Type'])
print("Cleaned Target Classes:\n", df['Accident_Type'].value_counts())

# Drop columns with >60% missing values
missing_pct = df.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 60].index.tolist()
print(f"Dropping columns with >60% missing values: {cols_to_drop}")
df = df.drop(columns=cols_to_drop)
df = df.drop(columns=['Accident_ID', 'Date'], errors='ignore')

# Fill missing values
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print("Missing values after cleaning:", df.isnull().sum().sum())

# Visualization: Accident type distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Accident_Type', data=df)
plt.title('Cleaned Accident Type Distribution')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
if not correlation_matrix.empty:
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    plt.show()

# Prepare features and target
y = df['Accident_Type']
X = df.drop(columns=['Accident_Type'])

# Remove outcome-dependent features (data leakage)
leakage_cols = [
    'Number_of_fatalities',
    'Number_of_sever_injuries',
    'Number_of_minor_injuries',
    'Number_of_Casualties'
]
X = X.drop(columns=leakage_cols, errors='ignore')
print(f"Leakage columns removed: {', '.join(leakage_cols)}")

# Encode target and features
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_encoded = pd.get_dummies(X, drop_first=True, dtype=int)
print(f"Features shape after encoding: {X_encoded.shape}")

# Split data: 70% train, 10% validation, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
)

print(f"Training: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"After SMOTE - Training shape: {X_train_resampled.shape}")

# Build neural network
input_dim = X_train_resampled.shape[1]
num_classes = len(le.classes_)

model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

# Compile and train
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

history = model.fit(
    X_train_resampled, y_train_resampled,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
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

# Evaluate on test set
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

target_names = [str(c) for c in le.classes_]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()