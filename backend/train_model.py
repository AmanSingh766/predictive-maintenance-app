import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

print("=" * 70)
print("🤖 INDUSTRIAL AI PREDICTIVE MAINTENANCE - MODEL TRAINING")
print("=" * 70)

# Check if dataset exists
dataset_path = '../dataset/predictive_maintenance.csv'
if not os.path.exists(dataset_path):
    print(f"\n❌ ERROR: Dataset file not found at {dataset_path}")
    print("⚠️  Please make sure predictive_maintenance.csv exists in the dataset folder!")
    sys.exit(1)

# Load dataset
print("\n📊 Loading Kaggle dataset...")
try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    sys.exit(1)

print("\n📋 Dataset Preview:")
print(df.head())
print(f"\n📊 Dataset Info:")
print(df.info())

# Check for missing values
print("\n🔍 Checking for missing values...")
missing = df.isnull().sum()
if missing.any():
    print("⚠️  Missing values found:")
    print(missing[missing > 0])
else:
    print("✅ No missing values found!")

# Data preprocessing
print("\n🔧 Preprocessing data...")

# Encode machine type (L, M, H)
le = LabelEncoder()
df['Type_Encoded'] = le.fit_transform(df['Type'])
print(f"✅ Machine types encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Select features for training
feature_cols = ['Air_temperature', 'Process_temperature', 'Rotational_speed', 
                'Torque', 'Tool_wear', 'Type_Encoded']

# Check if all columns exist
missing_cols = [col for col in feature_cols if col not in df.columns and col != 'Type_Encoded']
if missing_cols:
    print(f"❌ ERROR: Missing columns in dataset: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

X = df[feature_cols]
y = df['Machine_failure']

print(f"\n✅ Features selected: {feature_cols}")
print(f"\n📊 Target distribution:")
failure_count = (y == 1).sum()
no_failure_count = (y == 0).sum()
print(f"  No Failure (0): {no_failure_count} ({no_failure_count / len(y) * 100:.1f}%)")
print(f"  Failure (1): {failure_count} ({failure_count / len(y) * 100:.1f}%)")

# Handle imbalanced dataset warning
if failure_count < 10:
    print("\n⚠️  WARNING: Very few failure cases in dataset!")
    print("   Model might not predict failures accurately.")
    print("   Consider using more data or synthetic sampling techniques.")

# Split data
print("\n✂️ Splitting data (80% train, 20% test)...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
except ValueError as e:
    print(f"⚠️  Stratification failed, using simple split: {e}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Scale features
print("\n📐 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

# Train model
print("\n🎯 Training Random Forest Classifier...")
print("   Parameters:")
print("   - n_estimators: 100")
print("   - max_depth: 15")
print("   - min_samples_split: 5")
print("   - class_weight: balanced")
print("   - random_state: 42")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

print("\n⏳ Training in progress...")
model.fit(X_train_scaled, y_train)
print("✅ Model training completed!")

# Evaluate model
print("\n📈 Evaluating model performance...")
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'], zero_division=0))

print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n  True Negatives (Correct 'No Failure'): {cm[0][0]}")
print(f"  False Positives (Wrong 'Failure'): {cm[0][1]}")
print(f"  False Negatives (Missed 'Failure'): {cm[1][0]}")
print(f"  True Positives (Correct 'Failure'): {cm[1][1]}")

# Feature importance
print("\n🌟 Feature Importance Ranking:")
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"  {row['Feature']:25s} {bar} {row['Importance']:.4f}")

# Create models directory
print("\n💾 Saving trained model and preprocessing objects...")
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"✅ Created directory: {models_dir}/")

# Save all objects
try:
    joblib.dump(model, os.path.join(models_dir, 'predictive_model.pkl'))
    print(f"✅ Saved: {models_dir}/predictive_model.pkl")
    
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"✅ Saved: {models_dir}/scaler.pkl")
    
    joblib.dump(le, os.path.join(models_dir, 'label_encoder.pkl'))
    print(f"✅ Saved: {models_dir}/label_encoder.pkl")
    
    print("\n✅ All files saved successfully!")
except Exception as e:
    print(f"\n❌ Error saving files: {e}")
    sys.exit(1)

# Verify saved files
print("\n🔍 Verifying saved files...")
files_to_check = ['predictive_model.pkl', 'scaler.pkl', 'label_encoder.pkl']
all_exist = True
for filename in files_to_check:
    filepath = os.path.join(models_dir, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {filename:25s} ({size:,} bytes)")
    else:
        print(f"❌ {filename:25s} NOT FOUND!")
        all_exist = False

if not all_exist:
    print("\n⚠️  WARNING: Some files were not saved properly!")
    sys.exit(1)

# Test prediction with sample data
print("\n🧪 Testing model with sample prediction...")
try:
    sample = X_test.iloc[0:1]
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0]
    
    print(f"\nSample Input Values:")
    for col in feature_cols:
        if col == 'Type_Encoded':
            type_val = int(sample[col].values[0])
            type_name = le.inverse_transform([type_val])[0]
            print(f"  {col:25s}: {type_val} ({type_name})")
        else:
            print(f"  {col:25s}: {sample[col].values[0]:.2f}")
    
    print(f"\n🔮 Prediction Results:")
    print(f"  Prediction: {'⚠️  FAILURE' if prediction == 1 else '✅ NO FAILURE'}")
    print(f"  Failure Probability: {probability[1] * 100:.2f}%")
    print(f"  Healthy Probability: {probability[0] * 100:.2f}%")
    
    # Risk assessment
    risk = probability[1] * 100
    if risk < 30:
        risk_level = "🟢 LOW RISK"
    elif risk < 60:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🔴 HIGH RISK"
    
    print(f"  Risk Level: {risk_level}")
    
except Exception as e:
    print(f"⚠️  Could not perform test prediction: {e}")

# Final summary
print("\n" + "=" * 70)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📝 Summary:")
print(f"  ✅ Model trained with {len(X_train)} samples")
print(f"  ✅ Accuracy: {accuracy * 100:.2f}%")
print(f"  ✅ 3 files saved in '{models_dir}/' directory")
print("\n📌 Next Steps:")
print("  1. Run: python app.py")
print("  2. Open frontend/index.html in browser")
print("  3. Start making predictions!")
print("=" * 70 + "\n")