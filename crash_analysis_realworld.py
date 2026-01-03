"""
Neural Network Classification for Road Crash Data Analysis - REAL-WORLD VERSION
================================================================================

This is a modified version that excludes "outcome-dependent" features that
would not be known at the time of prediction in a real-world scenario.

EXCLUDED FEATURES (not available at prediction time):
- Number of fatalities
- Number of severe injuries  
- Number of minor injuries
- Number of Casualties
- Victim injury severity (Victim-1/2/3 Injury Severity)

This represents what the model could actually predict at an accident scene
BEFORE medical assessment.

Author: AI Senior Project
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("=" * 70)
print("REAL-WORLD NEURAL NETWORK - CRASH SEVERITY PREDICTION")
print("(Excluding outcome-dependent features)")
print("=" * 70)

# =============================================================================
# TASK 1: DATA LOADING
# =============================================================================
print("\n" + "=" * 70)
print("TASK 1: DATA LOADING")
print("=" * 70)

print("\n[1.1] Loading dataset...")
df = pd.read_excel('Crash Data AA.xlsx')
print(f"Dataset loaded successfully!")
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# TASK 2: DATA PREPROCESSING (REAL-WORLD VERSION)
# =============================================================================
print("\n" + "=" * 70)
print("TASK 2: DATA PREPROCESSING - REAL-WORLD VERSION")
print("=" * 70)

df_clean = df.copy()

# [2.1] Standardize target variable
print("\n[2.1] Standardizing target variable...")
pdo_variants = ['PDO', 'property damage only', 'Pdo', 'P.D.O', 'pdo', 'POD']
df_clean['Accident Type'] = df_clean['Accident Type'].replace(pdo_variants, 'PDO')
print(f"Target distribution:\n{df_clean['Accident Type'].value_counts()}")

# [2.2] Remove rows with missing target
print("\n[2.2] Removing rows with missing target...")
initial_rows = len(df_clean)
df_clean = df_clean.dropna(subset=['Accident Type'])
print(f"Removed {initial_rows - len(df_clean)} rows with missing target")
print(f"Remaining rows: {len(df_clean):,}")

# [2.3] Drop columns with 100% missing values
print("\n[2.3] Dropping columns with 100% missing values...")
cols_to_drop = ['Region', 'Zone', 'Exiting/entering', 'Contributory Action', 'Driving License']
cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
df_clean = df_clean.drop(columns=cols_to_drop)
print(f"Dropped columns: {cols_to_drop}")

# [2.4] Drop columns with >70% missing
print("\n[2.4] Dropping columns with >70% missing...")
high_missing_cols = []
for col in df_clean.columns:
    if df_clean[col].isnull().sum() / len(df_clean) > 0.70:
        high_missing_cols.append(col)

victim_cols = [col for col in df_clean.columns if 'Victim' in col and 'Movement' in col]
high_missing_cols.extend(victim_cols)
high_missing_cols = list(set(high_missing_cols))

df_clean = df_clean.drop(columns=high_missing_cols)
print(f"Dropped {len(high_missing_cols)} columns with high missing values")

# [2.5] *** CRITICAL: Drop outcome-dependent features (Real-World Exclusion) ***
print("\n" + "!" * 70)
print("REAL-WORLD EXCLUSION: Removing outcome-dependent features")
print("!" * 70)
print("\nThese features would NOT be known at prediction time:")

outcome_features = [
    'Number of fatalities',
    'Number of sever injuries',
    'Number of minor injuries',
    'Number of Casualties',
    'Victim-1 Injury Severity',
    'Victim-2 Injury Severity',
    'Victim-3 Injury Severity'
]

# Find which ones actually exist in the dataframe
existing_outcome_features = [col for col in outcome_features if col in df_clean.columns]
print(f"  - {', '.join(existing_outcome_features)}")

df_clean = df_clean.drop(columns=existing_outcome_features, errors='ignore')
print(f"\nDropped {len(existing_outcome_features)} outcome-dependent columns")
print(f"Remaining columns: {len(df_clean.columns)}")
print("!" * 70)

# [2.6] Drop identifier columns
print("\n[2.6] Dropping identifier columns...")
id_cols = ['Accident ID']
df_clean = df_clean.drop(columns=id_cols, errors='ignore')
print(f"Dropped: {id_cols}")

# [2.7] Handle remaining missing values
print("\n[2.7] Handling remaining missing values...")

numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()

if 'Accident Type' in categorical_cols:
    categorical_cols.remove('Accident Type')

# Fill numerical with median
for col in numerical_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        
# Fill categorical with mode or 'Unknown'
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()
        if len(mode_val) > 0:
            df_clean[col] = df_clean[col].fillna(mode_val[0])
        else:
            df_clean[col] = df_clean[col].fillna('Unknown')

print(f"Filled {len(numerical_cols)} numerical columns with median")
print(f"Filled {len(categorical_cols)} categorical columns with mode")

remaining_missing = df_clean.isnull().sum().sum()
print(f"Remaining missing values: {remaining_missing}")

# [2.8] Encode categorical variables
print("\n[2.8] Encoding categorical variables...")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le
    
print(f"Encoded {len(categorical_cols)} categorical columns using LabelEncoder")

# Encode target variable
target_encoder = LabelEncoder()
df_clean['Accident Type'] = target_encoder.fit_transform(df_clean['Accident Type'])
class_names = target_encoder.classes_
print(f"Target classes: {list(class_names)}")
print(f"Encoded as: {list(range(len(class_names)))}")

# [2.9] Prepare features and target
print("\n[2.9] Preparing features and target...")
X = df_clean.drop(columns=['Accident Type'])
y = df_clean['Accident Type']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeatures used ({X.shape[1]} total):")
for i, col in enumerate(X.columns, 1):
    print(f"  {i:2d}. {col}")

# [2.10] Feature scaling
print("\n[2.10] Applying feature scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Applied StandardScaler to all features")

# [2.11] Train-Test Split (70-30)
print("\n[2.11] Splitting data (70-30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:     {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Convert to numpy arrays for Keras
y_train = np.array(y_train)
y_test = np.array(y_test)

# One-hot encode targets
num_classes = len(class_names)
y_train_cat = keras.utils.to_categorical(y_train, num_classes)
y_test_cat = keras.utils.to_categorical(y_test, num_classes)

print(f"\nOne-hot encoded targets - shape: {y_train_cat.shape}")

# =============================================================================
# TASK 3: MODEL DESIGN
# =============================================================================
print("\n" + "=" * 70)
print("TASK 3: MODEL DESIGN")
print("=" * 70)

input_dim = X_train.shape[1]
output_dim = num_classes

print(f"\n[3.1] Neural Network Architecture:")
print("-" * 40)
print(f"""
Architecture Design:
====================
Input Layer:      {input_dim} features (REAL-WORLD ONLY - no outcome features)
                     ↓
Hidden Layer 1:   Dense(128, ReLU) + BatchNorm + Dropout(0.3)
                     ↓
Hidden Layer 2:   Dense(64, ReLU) + BatchNorm + Dropout(0.3)
                     ↓
Hidden Layer 3:   Dense(32, ReLU) + BatchNorm + Dropout(0.2)
                     ↓
Output Layer:     Dense({output_dim}, Softmax) → {list(class_names)}
""")

def build_model(input_dim, output_dim):
    """Build a neural network for multi-class classification."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Hidden Layer 1
        layers.Dense(128, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Hidden Layer 2
        layers.Dense(64, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        # Hidden Layer 3
        layers.Dense(32, kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        # Output Layer
        layers.Dense(output_dim, activation='softmax')
    ])
    
    return model

model = build_model(input_dim, output_dim)

print("\n[3.2] Model Summary:")
print("-" * 40)
model.summary()

# =============================================================================
# TASK 4: TRAINING PROCESS
# =============================================================================
print("\n" + "=" * 70)
print("TASK 4: TRAINING PROCESS")
print("=" * 70)

# Compute class weights
print("\n[4.1] Computing class weights...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Class weights:")
for i, (cls, weight) in enumerate(zip(class_names, class_weights)):
    print(f"  {cls}: {weight:.3f}")

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6,
    verbose=1
)

# Train the model
print("\n[4.2] Training the model...")
print("-" * 40)

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
print("\n[4.3] Plotting training history...")
import os
output_dir = os.path.dirname(os.path.abspath(__file__))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax1.set_title('Model Loss Over Epochs (Real-World)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Accuracy plot
ax2 = axes[1]
ax2.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax2.set_title('Model Accuracy Over Epochs (Real-World)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_history_realworld.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: training_history_realworld.png")
plt.close()

# =============================================================================
# TASK 5: EVALUATION
# =============================================================================
print("\n" + "=" * 70)
print("TASK 5: EVALUATION - REAL-WORLD MODEL")
print("=" * 70)

# Evaluate on test set
print("\n[5.1] Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Test Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions
print("\n[5.2] Making predictions...")
y_pred_proba = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification metrics
print("\n[5.3] Classification Report:")
print("-" * 70)
print(classification_report(y_test, y_pred, target_names=class_names))

# Get detailed metrics
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

print("\n" + "=" * 70)
print("DETAILED CLASSIFICATION METRICS - REAL-WORLD MODEL")
print("=" * 70)

print(f"\n{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
print("-" * 70)

for class_name in class_names:
    precision = report_dict[class_name]['precision']
    recall = report_dict[class_name]['recall']
    f1 = report_dict[class_name]['f1-score']
    support = int(report_dict[class_name]['support'])
    print(f"{class_name:<20} {precision:>12.2f} {recall:>12.2f} {f1:>12.2f} {support:>12}")

print("-" * 70)

accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"{'accuracy':<20} {'':<12} {'':<12} {accuracy:>12.2f} {len(y_test):>12}")
print(f"{'macro avg':<20} {precision_macro:>12.2f} {recall_macro:>12.2f} {f1_macro:>12.2f} {len(y_test):>12}")
print(f"{'weighted avg':<20} {precision_score(y_test, y_pred, average='weighted'):>12.2f} {recall_score(y_test, y_pred, average='weighted'):>12.2f} {f1_weighted:>12.2f} {len(y_test):>12}")
print("=" * 70)

# Confusion Matrix
print("\n[5.4] Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=class_names, yticklabels=class_names)
ax1.set_title('Confusion Matrix - Real-World Model (Counts)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')

ax2 = axes[1]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
            xticklabels=class_names, yticklabels=class_names)
ax2.set_title('Confusion Matrix - Real-World Model (Normalized)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_realworld.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: confusion_matrix_realworld.png")
plt.close()

# Per-class performance
print("\n[5.5] Creating per-class performance visualization...")
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['precision', 'recall', 'f1-score']
x = np.arange(len(class_names))
width = 0.25

for i, metric in enumerate(metrics):
    values = [report[cls][metric] for cls in class_names]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, values, width, label=metric.capitalize())

ax.set_xlabel('Accident Type')
ax.set_ylabel('Score')
ax.set_title('Per-Class Performance - Real-World Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'per_class_performance_realworld.png'), dpi=150, bbox_inches='tight')
print(f"  Saved: per_class_performance_realworld.png")
plt.close()

# Save model
print("\n[5.6] Saving trained model...")
model.save(os.path.join(output_dir, 'crash_classification_model_realworld.keras'))
print(f"  Saved: crash_classification_model_realworld.keras")

# =============================================================================
# FINAL COMPARISON & INTERPRETATION
# =============================================================================
print("\n" + "=" * 70)
print("REAL-WORLD MODEL PERFORMANCE ANALYSIS")
print("=" * 70)

print(f"""
REAL-WORLD MODEL RESULTS:
=========================
Test Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)
Macro F1-Score:   {f1_macro:.4f}
Weighted F1:      {f1_weighted:.4f}

FEATURES NOT USED (Outcome-Dependent):
======================================
- Number of fatalities
- Number of severe injuries
- Number of minor injuries
- Number of Casualties
- Victim injury severity

WHAT THIS MEANS:
================
This model represents what could be predicted at an accident scene
BEFORE detailed medical assessment. The performance drop compared to
the full model shows how much those outcome features were "cheating"
by already knowing the severity.

This is the more realistic and practical model for:
- Emergency response resource allocation
- Real-time severity prediction systems
- Automatic accident reporting systems

EXPECTED PERFORMANCE DIFFERENCE:
================================
The real-world model will likely have:
- Lower overall accuracy (expected drop of 10-20%)
- More difficulty with minority classes (Fatal, Serious)
- Better separation of PDO vs. Minor (since we can't "cheat")

This represents the TRUE predictive power of the available features!
""")

print("\n" + "=" * 70)
print("REAL-WORLD MODEL COMPLETED")
print("=" * 70)
