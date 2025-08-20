import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class DecisionTreeMLflowTracker:
    def __init__(self, experiment_name="melbourne_accident_severity"):
        """Initialize MLflow tracking"""
        # Set MLflow tracking URI (stores data locally)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        print(f"✅ MLflow experiment '{experiment_name}' is ready!")
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare Melbourne accident data"""
        print("📊 Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Original data shape: {df.shape}")
        
        # Select required features
        required_columns = ['ACCIDENT_DATE', 'ACCIDENT_TIME', 'DAY_WEEK_DESC', 'SPEED_ZONE', 'SEVERITY']
        
        # Check if all columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Filter to required columns
        df = df[required_columns].copy()
        
        # Handle missing values
        print(f"Missing values per column:")
        print(df.isnull().sum())
        
        # Drop rows with missing values
        df = df.dropna()
        print(f"Data shape after removing missing values: {df.shape}")
        
        return df
    
    def categorize_accident_time(self, accident_time):
        """Convert accident time to 4 categories: morning, afternoon, evening, late night"""
        # Extract hour 
        hour = accident_time.split(':')[0]
        
        # Return categories based on hours
        hour = int(hour)
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Late Night'
    
    def categorize_accident_date(self, accident_date):
        """Convert accident date to Australian seasons"""
        # Extract month as integer
        month = int(accident_date.split("-")[1])
        
        # Australian seasons:
        # Summer: Dec (12), Jan (1), Feb (2)
        # Autumn: Mar (3), Apr (4), May (5)
        # Winter: Jun (6), Jul (7), Aug (8)
        # Spring: Sep (9), Oct (10), Nov (11)
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'
    
    def feature_engineering(self, df):
        """Engineer features from raw data"""
        print("🔧 Engineering features...")
        
        df_processed = df.copy()
        
        # Convert ACCIDENT_DATE to Australian seasons
        print("📅 Converting dates to seasons...")
        df_processed['ACCIDENT_SEASON'] = df_processed['ACCIDENT_DATE'].apply(self.categorize_accident_date)
        
        # Convert ACCIDENT_TIME to time categories
        print("⏰ Converting times to categories...")
        df_processed['ACCIDENT_TIME_CATEGORY'] = df_processed['ACCIDENT_TIME'].apply(self.categorize_accident_time)
        
        # Encode all categorical variables
        le_day = LabelEncoder()
        le_season = LabelEncoder()
        le_time_cat = LabelEncoder()
        
        df_processed['DAY_WEEK_ENCODED'] = le_day.fit_transform(df_processed['DAY_WEEK_DESC'])
        df_processed['SEASON_ENCODED'] = le_season.fit_transform(df_processed['ACCIDENT_SEASON'])
        df_processed['TIME_CATEGORY_ENCODED'] = le_time_cat.fit_transform(df_processed['ACCIDENT_TIME_CATEGORY'])
        
        # Encode target variable
        le_severity = LabelEncoder()
        df_processed['SEVERITY_ENCODED'] = le_severity.fit_transform(df_processed['SEVERITY'])
        
        # Select final features (all categorical now)
        feature_columns = ['DAY_WEEK_ENCODED', 'SEASON_ENCODED', 'TIME_CATEGORY_ENCODED', 'SPEED_ZONE']
        X = df_processed[feature_columns]
        y = df_processed['SEVERITY_ENCODED']
        
        # Print feature mappings for clarity
        print(f"✅ Features prepared: {feature_columns}")
        print(f"📊 Feature mappings:")
        print(f"  Day of week: {dict(zip(le_day.classes_, range(len(le_day.classes_))))}")
        print(f"  Seasons: {dict(zip(le_season.classes_, range(len(le_season.classes_))))}")
        print(f"  Time categories: {dict(zip(le_time_cat.classes_, range(len(le_time_cat.classes_))))}")
        print(f"Target classes: {le_severity.classes_}")
        print(f"Class distribution:")
        print(df_processed['SEVERITY'].value_counts())
        
        # Print categorical distributions
        print(f"\n📈 Categorical distributions:")
        print(f"Seasons: {df_processed['ACCIDENT_SEASON'].value_counts()}")
        print(f"Time categories: {df_processed['ACCIDENT_TIME_CATEGORY'].value_counts()}")
        
        return X, y, le_severity
    
    def train_decision_tree(self, data_path):
        """Train Decision Tree with MLflow tracking"""
        
        # Load and prepare data
        df = self.load_and_prepare_data(data_path)
        if df is None:
            return None
        
        X, y, label_encoder = self.feature_engineering(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"decision_tree_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Hyperparameters for Decision Tree
            params = {
                'criterion': 'gini',
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': None,
                'random_state': 42
            }
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("features", X.columns.tolist())
            mlflow.log_param("target_classes", label_encoder.classes_.tolist())
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Initialize and train model
            print("🚀 Training Decision Tree...")
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, average='weighted')
            test_recall = recall_score(y_test, y_pred_test, average='weighted')
            test_f1 = f1_score(y_test, y_pred_test, average='weighted')
            
            # 5-Fold Cross Validation
            print("🔄 Running 5-fold cross validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
            mlflow.log_metric("test_precision", test_precision)
            mlflow.log_metric("test_recall", test_recall)
            mlflow.log_metric("test_f1_score", test_f1)
            mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
            mlflow.log_metric("cv_std_accuracy", cv_scores.std())
            
            # Log individual CV fold scores
            for i, score in enumerate(cv_scores):
                mlflow.log_metric(f"cv_fold_{i+1}_accuracy", score)
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Feature Importance:")
            print(feature_importance)
            
            # Save and log feature importance
            feature_importance.to_csv("feature_importance_dt.csv", index=False)
            mlflow.log_artifact("feature_importance_dt.csv")
            
            # Classification report
            class_report = classification_report(y_test, y_pred_test, 
                                               target_names=label_encoder.classes_,
                                               output_dict=True)
            
            # Log per-class metrics
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{class_name}_{metric_name}", value)
            
            # Create and save confusion matrix plot
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cm, annot=True, fmt='d', 
                       xticklabels=label_encoder.classes_,
                       yticklabels=label_encoder.classes_,
                       cmap='Blues')
            plt.title('Decision Tree - Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig("confusion_matrix_dt.png", dpi=300, bbox_inches='tight')
            mlflow.log_artifact("confusion_matrix_dt.png")
            plt.close()
            
            # Log the trained model
            mlflow.sklearn.log_model(
                model, 
                "decision_tree_model",
                input_example=X_test.iloc[:5]  # Log example input
            )
            
            # Print results
            print("\n" + "="*50)
            print("🎯 DECISION TREE RESULTS")
            print("="*50)
            print(f"Train Accuracy: {train_accuracy:.4f}")
            print(f"Test Accuracy:  {test_accuracy:.4f}")
            print(f"Test Precision: {test_precision:.4f}")
            print(f"Test Recall:    {test_recall:.4f}")
            print(f"Test F1-Score:  {test_f1:.4f}")
            print(f"CV Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"CV Scores: {cv_scores}")
            print("="*50)
            
            # Clean up temporary files
            try:
                os.remove("feature_importance_dt.csv")
                os.remove("confusion_matrix_dt.png")
            except:
                pass
            
            return model, {
                'test_accuracy': test_accuracy,
                'test_f1_score': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

def main():
    """Main function to run the experiment"""
    print("🌟 Starting Melbourne Accident Severity - Decision Tree Experiment")
    
    # Initialize tracker
    tracker = DecisionTreeMLflowTracker()
    
    # Path to your data file
    data_path = "datasets/merged.csv"  # Adjust this path!
    
    # Train model
    model, metrics = tracker.train_decision_tree(data_path)
    
    if model is not None:
        print("\n✅ Experiment completed successfully!")
        print("\n🖥️  To view results, run: mlflow ui")
        print("📱 Then open: http://localhost:5000")
    else:
        print("❌ Experiment failed - please check your data file")

if __name__ == "__main__":
    main()