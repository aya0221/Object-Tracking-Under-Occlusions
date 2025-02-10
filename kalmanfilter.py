import cv2
import numpy as np

class KalmanFilter:
    def __init__(self):
        # Initialize the Kalman Filter with 4 state variables and 2 measurement variables
        # 4 state variables: [x position, y position, x velocity, y velocity]
        # 2 measurement variables: [x position, y position]
        self.kf = cv2.KalmanFilter(4, 2)
        
        # Set the measurement matrix (relating measurement to state) [X position, Y position, X velocity, Y velocity]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],  # only the position in the X direction is measured and the velocity is not measured (so it's zero)
                                              [0, 1, 0, 0]], np.float32) # only the position in the Y direction is measured
        
        # Set the transition matrix (relates previous state to current state)
        # The system assumes constant velocity
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],  # The position depends on the previous position and velocity
                                              [0, 1, 0, 1],  # Same for y position
                                              [0, 0, 1, 0],  # The velocity is constant
                                              [0, 0, 0, 1]], np.float32)
        
        # Set the process noise covariance matrix (reflecting the uncertainty of the system)
        # A higher value here implies more uncertainty in the state
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05

    def visible_predict(self, coordX, coordY):
        """
        Use the Kalman Filter to predict the next state based on visible measurements (x, y coordinates).
        The Kalman Filter is corrected with the new measurement.
        
        Parameters:
            coordX (float): The x-coordinate measurement
            coordY (float): The y-coordinate measurement
            
        Returns:
            tuple: Predicted x and y coordinates as integers

        # Correct with new measurement (center of bounding box) → Predict again (next position)
        """
        # Create a measurement array with the visible coordinates
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        
        # Correct the Kalman Filter with the measurement (updates the state estimate)
        self.kf.correct(measured)
        
        # Predict the next state (x, y, velocity in x, velocity in y)
        predicted = self.kf.predict()
        
        # Return the predicted x and y positions as integers
        x, y = int(predicted[0]), int(predicted[1])
        return x, y

    def hidden_predict(self):
        """
        Predict the next state when the object is not visible. 
        Uses the previous state and system model to predict the next position.
        
        Returns:
            tuple: Predicted x and y coordinates as integers
        """
        # Predict the next state (x, y, velocity in x, velocity in y)
        predicted = self.kf.predict()
        
        # Extract the predicted x and y coordinates
        x, y = int(predicted[0]), int(predicted[1])
        
        # Correct the state estimate based on the predicted state (this can be used to improve future predictions)
        self.kf.correct(np.array([[np.float32(x)], [np.float32(y)]]))
        
        # Return the predicted x and y coordinates as integers
        return x, y
