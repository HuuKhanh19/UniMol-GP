"""
UniMol Model Wrapper

Wrapper for UniMol-tools v1 to provide a consistent interface.
This will be extended in Step 2 (EGGROLL) and Step 3 (GP).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import mean_squared_error, roc_auc_score


class UniMolWrapper:
    """
    Wrapper for UniMol-tools molecular property prediction.
    
    This class provides a consistent interface for training and evaluation.
    It wraps the UniMol MolTrain and MolPredict classes.
    """
    
    def __init__(
        self,
        task_type: str = "regression",
        use_gpu: bool = True,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        early_stopping_patience: int = 10,
        save_path: str = "./experiments",
        random_seed: int = 42
    ):
        """
        Initialize UniMol wrapper.
        
        Args:
            task_type: 'regression' or 'classification'
            use_gpu: Whether to use GPU
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            save_path: Path to save models and results
            random_seed: Random seed
        """
        self.task_type = task_type
        self.use_gpu = use_gpu
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.save_path = save_path
        self.random_seed = random_seed
        
        self.model = None
        self.is_trained = False
        
    def train(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "target"
    ) -> Dict[str, Any]:
        """
        Train UniMol model.
        
        Args:
            train_data: Training DataFrame
            valid_data: Validation DataFrame (used for monitoring only, not for training)
            smiles_column: Name of SMILES column
            target_column: Name of target column
            
        Returns:
            Training results dictionary
        """
        from unimol_tools import MolTrain
        import os
        
        # Only use train data for training (no k-fold)
        train_data = train_data.copy()
        
        # Rename columns to match UniMol expected format
        train_data = train_data.rename(columns={
            smiles_column: 'SMILES',
            target_column: 'TARGET'
        })
        
        # Save to temporary CSV file
        os.makedirs(self.save_path, exist_ok=True)
        temp_csv_path = os.path.join(self.save_path, 'train_data.csv')
        train_data.to_csv(temp_csv_path, index=False)
        
        # Create MolTrain instance
        # kfold=1 means no cross-validation, all data used for training
        # No early stopping with kfold=1, so we use fixed epochs
        trainer = MolTrain(
            task=self.task_type,
            data_type='molecule',
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            early_stopping=self.early_stopping_patience,  # Won't work with kfold=1
            metrics='mse' if self.task_type == 'regression' else 'auc',
            split='random',  # Doesn't matter with kfold=1
            kfold=1,  # No cross-validation, train on all data
            save_path=self.save_path,
            use_cuda=self.use_gpu,
            remove_hs=True,
            smiles_col='SMILES',
            target_cols='TARGET'
        )
        
        # Fit the model with CSV file path
        trainer.fit(temp_csv_path)
        
        self.model = trainer
        self.is_trained = True
        
        return {"status": "trained", "save_path": self.save_path}
    
    def predict(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles"
    ) -> np.ndarray:
        """
        Make predictions with trained model.
        
        Args:
            data: DataFrame with SMILES
            smiles_column: Name of SMILES column
            
        Returns:
            Predictions array
        """
        from unimol_tools import MolPredict
        import os
        
        # Prepare data - save to temp CSV
        temp_data = data[[smiles_column]].copy()
        temp_data.columns = ['SMILES']
        
        temp_csv_path = os.path.join(self.save_path, 'predict_data.csv')
        temp_data.to_csv(temp_csv_path, index=False)
        
        # Load predictor and predict (only model_0 exists with kfold=1)
        predictor = MolPredict(load_model=self.save_path)
        predictions = predictor.predict(temp_csv_path)
        
        # predictions is a dict with 'predict' key
        if isinstance(predictions, dict):
            pred_values = predictions.get('predict', predictions)
        else:
            pred_values = predictions
            
        return np.array(pred_values).flatten()
    
    def evaluate(
        self,
        data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "target"
    ) -> Dict[str, float]:
        """
        Evaluate model on data.
        
        Args:
            data: DataFrame with SMILES and targets
            smiles_column: SMILES column name
            target_column: Target column name
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(data, smiles_column)
        targets = data[target_column].values
        
        results = {}
        
        if self.task_type == "regression":
            mse = mean_squared_error(targets, predictions)
            rmse = np.sqrt(mse)
            results["mse"] = mse
            results["rmse"] = rmse
        else:
            auc = roc_auc_score(targets, predictions)
            results["auc"] = auc
            
        return results
    
    def save(self, path: Optional[str] = None):
        """Save model (handled by UniMol internally)."""
        if path:
            print(f"Model saved at: {path}")
        else:
            print(f"Model saved at: {self.save_path}")
            
    def load(self, path: str):
        """Load trained model for prediction."""
        self.save_path = path
        self.is_trained = True


class Step1Trainer:
    """
    Step 1 Trainer: Baseline UniMol with Gradient Descent.
    
    This is the standard training procedure using UniMol's default
    gradient descent optimization.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        experiment_name: str = "step1_baseline"
    ):
        """
        Initialize Step 1 trainer.
        
        Args:
            config: Configuration dictionary
            experiment_name: Name for this experiment
        """
        self.config = config
        self.experiment_name = experiment_name
        
        # Extract settings
        task_type = config['dataset']['task_type']
        
        # Create output directory
        output_dir = os.path.join(
            config['experiment']['output_dir'],
            experiment_name,
            config['dataset']['name']
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize wrapper
        self.wrapper = UniMolWrapper(
            task_type=task_type,
            use_gpu=config['unimol']['use_gpu'],
            epochs=config['training']['epochs'],
            batch_size=config['training']['batch_size'],
            learning_rate=config['training']['learning_rate'],
            early_stopping_patience=config['training']['early_stopping_patience'],
            save_path=output_dir,
            random_seed=config['data']['random_seed']
        )
        
        self.output_dir = output_dir
        self.task_type = task_type
        self.metric_name = config['dataset']['metric']
        
    def run(
        self,
        train_data: pd.DataFrame,
        valid_data: pd.DataFrame,
        test_data: pd.DataFrame,
        smiles_column: str = "smiles",
        target_column: str = "measured"
    ) -> Dict[str, Any]:
        """
        Run complete training and evaluation pipeline.
        
        Args:
            train_data: Training DataFrame
            valid_data: Validation DataFrame
            test_data: Test DataFrame
            smiles_column: SMILES column name
            target_column: Target column name
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*60}")
        print(f"Step 1: Baseline Training (Gradient Descent)")
        print(f"Dataset: {self.config['dataset']['name']}")
        print(f"Task: {self.task_type}")
        print(f"Metric: {self.metric_name}")
        print(f"{'='*60}\n")
        
        # Train
        print("Training...")
        self.wrapper.train(
            train_data, 
            valid_data,
            smiles_column=smiles_column,
            target_column=target_column
        )
        
        # Evaluate on all splits
        print("\nEvaluating...")
        train_results = self.wrapper.evaluate(
            train_data, smiles_column, target_column
        )
        valid_results = self.wrapper.evaluate(
            valid_data, smiles_column, target_column
        )
        test_results = self.wrapper.evaluate(
            test_data, smiles_column, target_column
        )
        
        # Compile results
        results = {
            "dataset": self.config['dataset']['name'],
            "task_type": self.task_type,
            "metric": self.metric_name,
            "train": train_results,
            "valid": valid_results,
            "test": test_results,
            "model_path": self.output_dir
        }
        
        # Print results
        print(f"\n{'='*60}")
        print("Results:")
        print(f"  Train {self.metric_name.upper()}: {train_results.get(self.metric_name, 'N/A'):.4f}")
        print(f"  Valid {self.metric_name.upper()}: {valid_results.get(self.metric_name, 'N/A'):.4f}")
        print(f"  Test  {self.metric_name.upper()}: {test_results.get(self.metric_name, 'N/A'):.4f}")
        print(f"{'='*60}\n")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        import json
        
        results_path = os.path.join(self.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_path}")