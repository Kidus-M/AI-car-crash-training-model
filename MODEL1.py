"""
Neural Network Classification for Road Crash Data Analysis with consideration of number of casualties
Complete ML pipeline for accident severity classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

output_dir = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("NEURAL NETWORK CLASSIFICATION FOR ROAD CRASH DATA")
print("=" * 70)

# ============================================================================
# DATA LOADING & EDA
# ============================================================================
print("\n[1] Loading dataset...")
df = pd.read_excel('Crash Data AA.xlsx')
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Missing values analysis
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct}).sort_values('Missing %', ascending=False)
print(f"\nColumns with missing values: {len(missing_df[missing_df['Missing %'] > 0])}")

# Target variable
print("\n[2] Target Variable (Accident Type):")
print(df['Accident Type'].value_counts())

# Visualizations
print("\n[3] Creating visualizations...")

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df_temp = df.copy()
pdo_variants = ['PDO', 'property damage only', 'Pdo', 'P.D.O', 'pdo', 'POD']
df_temp['Accident Type'] = df_temp['Accident Type'].replace(pdo_variants, 'PDO')

ax1 = axes[0]
target_counts = df['Accident Type'].value_counts()
bars = ax1.bar(range(len(target_counts)), target_counts.values, color=plt.cm.Set3(np.linspace(0, 1, len(target_counts))))
ax1.set_xticks(range(len(target_counts)))
ax1.set_xticklabels(target_counts.index, rotation=45, ha='right')
ax1.set_title('Before Standardization', fontweight='bold')
ax1.set_ylabel('Count')

ax2 = axes[1]
target_clean = df_temp['Accident Type'].value_counts()
bars2 = ax2.bar(target_clean.index, target_clean.values, color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
ax2.set_title('After Standardization', fontweight='bold')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_target_distribution.png'), dpi=150, bbox_inches='tight')
print("  Saved: eda_target_distribution.png")
plt.close()

# Numerical features
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
key_numerical = ['Driver age', 'Driver experiance(years)', 'Number of fatalities',
                 'Number of sever injuries', 'Number of minor injuries', 
                 'Number of involved vehicles', 'Estimated Property damage']
key_numerical = [col for col in key_numerical if col in numerical_cols]

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(key_numerical[:8]):
    data = df[col].dropna()
    axes[i].hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    axes[i].set_title(col, fontweight='bold')
    axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
    axes[i].legend(fontsize=8)
for j in range(len(key_numerical), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Numerical Features Distribution', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_numerical_distributions.png'), dpi=150, bbox_inches='tight')
print("  Saved: eda_numerical_distributions.png")
plt.close()

# Correlation matrix
key_corr = ['Driver age', 'Driver experiance(years)', 'Number of fatalities',
            'Number of sever injuries', 'Number of minor injuries', 'Number of involved vehicles', 
            'Estimated Property damage', 'Number of Casualties', 'Veh Year of Service']
key_corr = [col for col in key_corr if col in df.columns]

fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df[key_corr].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
ax.set_title('Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'eda_correlation_matrix.png'), dpi=150, bbox_inches='tight')
print("  Saved: eda_correlation_matrix.png")
plt.close()

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n[4] Data Preprocessing...")
df_clean = df.copy()

# Standardize target
pdo_variants = ['PDO', 'property damage only', 'Pdo', 'P.D.O', 'pdo', 'POD']
df_clean['Accident Type'] = df_clean['Accident Type'].replace(pdo_variants, 'PDO')
df_clean = df_clean.dropna(subset=['Accident Type'])
print(f"  Rows after removing missing target: {len(df_clean):,}")

# Drop high missing and ID columns
cols_to_drop = ['Region', 'Zone', 'Exiting/entering', 'Contributory Action', 'Driving License']
cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
df_clean = df_clean.drop(columns=cols_to_drop)

high_missing_cols = []
for col in df_clean.columns:
    if df_clean[col].isnull().sum() / len(df_clean) > 0.70:
        high_missing_cols.append(col)
victim_cols = [col for col in df_clean.columns if 'Victim' in col and 'Movement' in col]
high_missing_cols.extend(victim_cols)
high_missing_cols = list(set(high_missing_cols))
df_clean = df_clean.drop(columns=high_missing_cols)

df_clean = df_clean.drop(columns=['Accident ID'], errors='ignore')
print(f"  Remaining columns: {len(df_clean.columns)}")

# Fill missing values
numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
if 'Accident Type' in categorical_cols:
    categorical_cols.remove('Accident Type')

for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()
        df_clean[col] = df_clean[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

print(f"  Missing values after filling: {df_clean.isnull().sum().sum()}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
df_clean['Accident Type'] = target_encoder.fit_transform(df_clean['Accident Type'])
class_names = target_encoder.classes_
print(f"  Target classes: {list(class_names)}")

# Prepare features and target
X = df_clean.drop(columns=['Accident Type'])
y = df_clean['Accident Type']
print(f"  Features: {X.shape}, Target: {y.shape}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)
print(f"  Training: {X_train.shape[0]:,} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test: {X_test.shape[0]:,} ({X_test.shape[0]/len(X)*100:.1f}%)")

# One-hot encode targets
y_train = np.array(y_train)
y_test = np.array(y_test)
num_classes = len(class_names)
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

# ============================================================================
# MODEL DESIGN
# ============================================================================
print("\n[5] Building neural network...")
input_dim = X_train.shape[1]
output_dim = num_classes

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(64, kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(32, kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.2),
    layers.Dense(output_dim, activation='softmax')
])

model.summary()

# ============================================================================
# TRAINING
# ============================================================================
print("\n[6] Training model...")

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train_cat,
    validation_split=0.15,
    epochs=100,
    batch_size=64,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Loss Over Epochs', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Accuracy Over Epochs', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
print("  Saved: training_history.png")
plt.close()

print(f"\n  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[7] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")

y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Metrics
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"\nKey Metrics:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Macro F1: {f1_macro:.4f}")
print(f"  Weighted F1: {f1_weighted:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
print("  Saved: confusion_matrix.png")
plt.close()

# Per-class performance
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(class_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in class_names]
    offset = (i - 1) * width
    ax.bar(x + offset, values, width, label=metric.capitalize())

ax.set_xlabel('Accident Type')
ax.set_ylabel('Score')
ax.set_title('Per-Class Performance', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=150, bbox_inches='tight')
print("  Saved: per_class_performance.png")
plt.close()

# Save model
model.save(os.path.join(output_dir, 'crash_classification_model.keras'))
print("  Saved: crash_classification_model.keras")

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)
print(f"\nTest Accuracy: {accuracy:.4f}")
print(f"Macro F1: {f1_macro:.4f}")
print(f"Weighted F1: {f1_weighted:.4f}")
