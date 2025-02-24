from collections import deque
from abc import ABC, abstractmethod

import numpy as np

# example usage
# running_pitch = RunningCalculator(strategy=ExponentialAverageStrategy(alpha=0.3))

# TODO: check if normals need to be normalized (before and after updating)
# TODO: check other Kalman filter implementations (IEKF, UKF, etc.), without calibration?

class RunningCalculator:
    """
    A class that wraps a strategy for calculating any running calculation 
    (e.g. average, Kalman filter, etc.) Every strategy needs to implement
    the `update()` and `get_current_value()` methods.
    """
    def __init__(self, strategy):
        self.strategy = strategy

    def update(self, new_value):
        if isinstance(new_value, np.ndarray):
            norm = np.linalg.norm(new_value)
            if norm > 0:
                new_value = new_value / norm
            else:
                pass
        self.strategy.update(new_value)

    def get_current_value(self):
        current = self.strategy.get_current_value()
        if isinstance(current, np.ndarray):
            norm = np.linalg.norm(current)
            if norm > 0:
                return current / norm
            else:
                print("[WARN] Attempted to normalize a zero vector."
                      + "Assigning default unit vector.")
                return np.array([0.0, 0.0, 1.0])
        else:
            return current


class StrategyInterface(ABC):
    def __init__(self, n):
        self.n = n
        self.values = deque(maxlen=n)  # all strategies have `values`

    @abstractmethod
    def update(self, new_value):
        """Update the strategy with a new value."""
        pass

    @abstractmethod
    def get_current_value(self):
        """Retrieve the current calculated value."""
        pass


class IdentityStrategy(StrategyInterface):
    def __init__(self, n=5):
        self.n = n
        self.values = deque(maxlen=n)

    def update(self, new_value):
        self.values.append(new_value)

    def get_current_value(self):
        return self.values[-1] if self.values else None


class AverageStrategy(StrategyInterface):
    def __init__(self, n=5):
        self.n = n
        self.values = deque(maxlen=n)
        self.current_value = None

    def update(self, new_value):
        self.values.append(new_value)
        self.current_value = sum(self.values) / len(self.values)

    def get_current_value(self):
        return self.current_value


class ExponentialAverageStrategy(StrategyInterface):
    def __init__(self, n=5, alpha=0.5):
        self.n = n
        self.values = deque(maxlen=n)
        self.alpha = alpha
        self.current_value = None

    def update(self, new_value):
        self.values.append(new_value)
        # recalculate exponential average
        if len(self.values) == 1:
            self.current_value = new_value
        else:
            self.current_value = self.alpha * new_value + (1 - self.alpha) * self.current_value

    def get_current_value(self):
        return self.current_value


class BasicKalmanFilterStrategy(StrategyInterface):
    def __init__(self, n=5, process_noise=1.0, measurement_noise=1.0, huber_delta=1.0, burn_in=5):
        super().__init__(n)
        self.current_value = None
        self.P = None
        self.Q = process_noise
        self.R = measurement_noise
        self.huber_delta = huber_delta
        self.burn_in = burn_in
        self.init_measurements = []  # Store initial measurements

    def update(self, new_value):
        new_value = np.array(new_value)
        
        # Use a burn-in period to initialize the state
        if len(self.init_measurements) < self.burn_in:
            self.init_measurements.append(new_value)
            self.current_value = np.mean(self.init_measurements, axis=0)
            # Optionally, set a low initial covariance if you're confident in these measurements
            if new_value.ndim == 0:
                self.P = 10.0
            else:
                self.P = 10.0 * np.eye(new_value.shape[0])
            self.values.append(new_value)
            return
        
        # Proceed with the regular update once burn-in is done:
        if self.current_value is None:
            # This branch should not really be reached due to the burn-in
            self.current_value = new_value
            if new_value.ndim == 0:
                self.P = 10.0
            else:
                self.P = 10.0 * np.eye(new_value.shape[0])
        else:
            x_pred = self.current_value
            if np.isscalar(self.P):
                P_pred = self.P + self.Q
            else:
                P_pred = self.P + (self.Q * np.eye(self.current_value.shape[0]))
            
            if np.isscalar(P_pred):
                K = P_pred / (P_pred + self.R)
            else:
                R_matrix = (self.R * np.eye(self.current_value.shape[0])
                            if np.isscalar(self.R)
                            else self.R)
                K = P_pred @ np.linalg.inv(P_pred + R_matrix)
            
            innovation = new_value - x_pred
            if np.isscalar(innovation) or (isinstance(innovation, np.ndarray) and innovation.ndim == 0):
                innov_mag = abs(innovation)
            else:
                innov_mag = np.linalg.norm(innovation)
            
            if innov_mag <= self.huber_delta:
                weight = 1.0
            else:
                weight = self.huber_delta / innov_mag
            
            adjusted_innovation = innovation * weight
            
            if np.isscalar(x_pred) or (isinstance(x_pred, np.ndarray) and x_pred.ndim == 0):
                self.current_value = x_pred + K * adjusted_innovation
            else:
                self.current_value = x_pred + K @ adjusted_innovation

            if np.isscalar(P_pred):
                self.P = (1 - K) * P_pred
            else:
                I = np.eye(self.current_value.shape[0])
                self.P = (I - K) @ P_pred
        
        self.values.append(new_value)
    
    def get_current_value(self):
        return self.current_value



class RobustFilterStrategy(StrategyInterface):
    def __init__(self, window_size=5, threshold=0.1, alpha=0.5):
        """
        Args:
            window_size (int): Number of recent measurements to keep.
            threshold (float): Allowed difference between the new measurement and the median.
                               For vectors, the difference is computed via the Euclidean norm.
            alpha (float): Smoothing factor for exponential moving average (0 < alpha <= 1).
        """
        super().__init__(window_size)
        self.window_size = window_size
        self.threshold = threshold
        self.alpha = alpha
        self.window = deque(maxlen=window_size)
        self.filtered_value = None

    def update(self, new_value):
        # Convert measurement to a numpy array (if it isn’t already) for consistent processing.
        new_value = np.array(new_value)
        self.window.append(new_value)

        # Compute the median over the window.
        window_arr = np.array(self.window)
        median_value = np.median(window_arr, axis=0)

        # Calculate the difference between the new measurement and the median.
        diff = new_value - median_value
        if np.isscalar(diff) or diff.ndim == 0:
            diff_magnitude = abs(diff)
        else:
            diff_magnitude = np.linalg.norm(diff)

        # Outlier detection: if the new value is too far from the median,
        # we consider it an outlier and skip updating.
        if diff_magnitude > self.threshold:
            # Optionally, you might blend less or log this event. Here we simply ignore the update.
            self.values.append(new_value)
            return

        # Update the filtered value using exponential moving average.
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value

        self.values.append(new_value)

    def get_current_value(self):
        return self.filtered_value
