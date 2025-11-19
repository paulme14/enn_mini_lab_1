"""
Linear regression model using Gradient Descent (Task 4)

This file provides
    - A class interface (GradientDescentLinearModel) for extension
      and a basic structure for the implementation.
    - we also include a normalization of the data
        This is required as we have features of quite different magnitude
        which leads to instabilities. Normalization is provided.
    - the functions train_baseline_model(), evaluate_model() and continue_training()
      used in automated tests and notebooks.

This implementation uses full-batch Gradient Descent and can track learning curves.

You are tasked with implementing the model!
"""
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

class GradientDescentLinearModel:
    """
    Dummy version of a gradient-descent-based linear regression model.
    Students must implement: 
        - the gradient update in `_step`
        - the training loop inside `fit`
        - the continuation of training in `continue_training`
    """

    def __init__(self, learning_rate=0.005, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0.0
        # Lists for tracking of losses 
        self.train_curve = []
        self.val_curve = []
        self.mean_ = None
        self.std_ = None
        self.y_mean_ = 0.0

    # --------------------------------------------------
    # Data preprocessing
    # --------------------------------------------------
    def _scale(self, X):
        """Standardize input features."""
        return (X - self.mean_) / self.std_

    def _predict_raw(self, X):
        """Linear model forward pass (no activation)."""
        return X @ self.w + self.b

    # --------------------------------------------------
    # One training step (dummy!)
    # --------------------------------------------------
    def _step(self, Xs, ys):
        """
        Perform ONE gradient descent update.
        TODO: 
            - Compute predictions
            - Compute errors
            - Compute gradients for w and b
            - Update parameters using the learning rate
        """

        # Currently does nothing except compute a dummy RMSE
        pred = self._predict_raw(Xs)
        rmse = 0.
        # (Currently no update -> w and b never change)
        return rmse

    # --------------------------------------------------
    # Fit model to training data (dummy!)
    # --------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None):
        """Main training loop. You have to implement the update inside the loop."""
        # Save statistics - required for later predictions
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8

        # Scaling of inputs
        Xs = self._scale(X)
        ys = y - self.y_mean_

        # Same for validation data (if provided)
        if X_val is not None:
            Xs_val = self._scale(X_val)
            ys_val = y_val - self.y_mean_

        # Initialize weights
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0

        # Training loop
        for _ in range(self.epochs):

            # TODO: call the gradient-descent step
            tr_rmse = self._step(Xs, ys)

            # Next lines only collect training and validation losses
            self.train_curve.append(tr_rmse)

            # Validation RMSE (model does not improve yet)
            if X_val is not None:
                val_pred = self._predict_raw(Xs_val)
                val_rmse = float(np.sqrt(np.mean((val_pred - ys_val)**2)))
                self.val_curve.append(val_rmse)

        return self

    # --------------------------------------------------
    # Optional continuation of training (dummy!)
    # --------------------------------------------------
    def continue_training(self, X, y, X_val=None, y_val=None, steps=500):
        """
        Continue training from existing weights.

        TODO:
            - Implement continued learning as in fit() above
        """
        Xs = self._scale(X)
        ys = y - self.y_mean_

        if X_val is not None:
            Xs_val = self._scale(X_val)
            ys_val = y_val - self.y_mean_

        for _ in range(steps):

            # TODO: add training step and adjusting weights
            tr_rmse = self._step(Xs, ys) 
            self.train_curve.append(tr_rmse)

            if X_val is not None:
                val_pred = self._predict_raw(Xs_val)
                val_rmse = float(np.sqrt(np.mean((val_pred - ys_val)**2)))
                self.val_curve.append(val_rmse)

        return self
    
    # --------------------------------------------------
    # Optional: You might as well want to have a
    # transfer training function (dummy!)
    # --------------------------------------------------

    def transfer_training(self,
                                X_pre_train, y_pre_train, X_pre_val, y_pre_val,
                                X_ft_train, y_ft_train, X_ft_val, y_ft_val,
                                fine_tune_steps=300,
                                prefix_pre="pre",
                                prefix_ft="ft"):
        """
        Transfer learning method using gradient descent:
        1) Pretrain on (X_pre)
        2) Evaluate before fine-tuning
        3) Fine-tune on (X_ft)
        4) Evaluate after fine-tuning
        Prefixes define naming in the result dict.
        """
        res = {}
        # ---- 1) Pretrain ----
        self.fit(X_pre_train, y_pre_train, X_pre_val, y_pre_val)

        # ---- 2) Evaluate before fine-tuning ----
        res[f"before_{prefix_pre}_{prefix_pre}"] = \
            self.evaluate(X_pre_train, y_pre_train, X_pre_val, y_pre_val)

        res[f"before_{prefix_pre}_{prefix_ft}"] = \
            self.evaluate(X_pre_train, y_pre_train, X_ft_val, y_ft_val)

        # ---- 3) Using the optional Fine-tune above ----
        self.continue_training(
            X_ft_train, y_ft_train,
            X_ft_val,   y_ft_val,
            steps=fine_tune_steps
        )

        # ---- 4) Evaluate after fine-tuning ----
        res[f"after_{prefix_ft}_{prefix_pre}"] = \
            self.evaluate(X_pre_train, y_pre_train, X_pre_val, y_pre_val)

        res[f"after_{prefix_ft}_{prefix_ft}"] = \
            self.evaluate(X_ft_train, y_ft_train, X_ft_val, y_ft_val)

        return res

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    def predict(self, X):
        """Return predictions (unscaled → scaled → model output)."""
        Xs = self._scale(X)
        return self._predict_raw(Xs) + self.y_mean_

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    def evaluate(self, X_train, y_train, X_val=None, y_val=None):
        """Evaluate model on train (and val) data."""
        y_pred_train = self.predict(X_train)
        result = {
            "r2_train": float(r2_score(y_train, y_pred_train)),
            "rmse_train": float(root_mean_squared_error(y_train, y_pred_train))
        }

        if X_val is not None:
            y_pred_val = self.predict(X_val)
            result["r2_val"] = float(r2_score(y_val, y_pred_val))
            result["rmse_val"] = float(root_mean_squared_error(y_val, y_pred_val))

        return result
