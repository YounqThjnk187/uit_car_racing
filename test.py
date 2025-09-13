from client_lib import GetStatus, GetSeg, AVControl, CloseSocket
import cv2
import numpy as np
import time

# Constants
CHECKPOINT1 = 160
CHECKPOINT2 = 140
CHECKPOINT3 = 120
MAX_ANGLE = 25
MAGIC_ANGLE = 10
SPEED = 32
MIN_SPEED = 1
WHITE_COLOR_VALUE = 255
BIAS_FACTOR = 0.6
LINE_THICKNESS = 2
INPUT_RANGE = (-5, 5)  # Input range of values
ANGLE_RANGE = (-25, 25)  # Desired angle range

# PID constants for angle control
KP_ANGLE = 1.3
KI_ANGLE = 0.02
KD_ANGLE = 0.1

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, current_value):
        error = setpoint - current_value
        self.integral += error
        # Anti-windup check
        if abs(error) < 1:  # Threshold for integral action
            self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

angle_pid = PID(KP_ANGLE, KI_ANGLE, KD_ANGLE)

def show_image(gray_image, centers, avg_slope):
    h, w = gray_image.shape

    for checkpoint, center_row in zip([CHECKPOINT1, CHECKPOINT2, CHECKPOINT3], centers):
        gray_image = cv2.line(gray_image, (0, checkpoint), (w, checkpoint), 90, LINE_THICKNESS)
        gray_image = cv2.line(gray_image, (center_row, checkpoint), (int(w/2), h - 1), 90, LINE_THICKNESS)

    # Draw the calculated angle line
    x1_angle = int(w / 2 + (h - CHECKPOINT1) * avg_slope)
    gray_image = cv2.line(gray_image, (int(w / 2), h), (x1_angle, CHECKPOINT1), 0, LINE_THICKNESS + 5)

    cv2.imshow('test', gray_image)

def get_center_row(gray, checkpoint):
    line_row = gray[checkpoint, :]
    line = np.where(line_row == WHITE_COLOR_VALUE)[0]
    if len(line) == 0:
        return int(gray.shape[1] / 2)  # Default to center if no line detected
    min_x, max_x = line[0], line[-1]
    # Handling crossroad
    if max_x - min_x >= gray.shape[1] - 1:
        return int(gray.shape[1] / 2) 
    return int((max_x + min_x + 1) * BIAS_FACTOR)

def is_crossroad(gray, checkpoint):
    line_row = gray[checkpoint, :]  # Fixed the syntax error here
    line = np.where(line_row == WHITE_COLOR_VALUE)[0]
    if len(line) != 0:
        min_x, max_x = line[0], line[-1]
        if max_x - min_x >= gray.shape[1] - 1:
            return True
    return False

# Function to calculate linear angle
def calculate_linear_angle(value, input_range, angle_range):
    input_min, input_max = input_range
    angle_min, angle_max = angle_range
    # Linear transformation formula
    linear_angle = (value - input_min) * (angle_max - angle_min) / (input_max - input_min) + angle_min
    return linear_angle

def calculate_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = (gray * (WHITE_COLOR_VALUE / np.max(gray))).astype(np.uint8)

    h, w = gray.shape
    checkpoints = [CHECKPOINT1, CHECKPOINT2, CHECKPOINT3]
    centers = [get_center_row(gray, cp) for cp in checkpoints]

    slopes = []
    for x1, y1 in zip(centers, checkpoints):
        x0, y0 = int(w / 2), h
        slope = (x1 - x0) / (y0 - y1)
        slopes.append(slope)
    
    # Average the slopes
    avg_slope = sum(slopes) / len(slopes)
    linear_angle = calculate_linear_angle(avg_slope, INPUT_RANGE, ANGLE_RANGE)

    show_image(gray, centers, avg_slope)

    return linear_angle

def calculate_speed(angle):
    speed = SPEED - abs(angle) * (SPEED - MIN_SPEED) / MAX_ANGLE
    speed = max(MIN_SPEED, speed)  # Ensure speed is not below MIN_SPEED
    if abs(angle) > MAGIC_ANGLE:
        speed = MIN_SPEED
    print(f"Speed: {speed:.2f}")
    return speed

if __name__ == "__main__":
    try:
        while True:
            time.sleep(0.05)  # Adjust sleep time for desired frame rate

            state = GetStatus()
            segment_image = GetSeg()

            if segment_image is None:
                print("Failed to get the segmented image.")
                continue

            angle = calculate_angle(segment_image)
            if angle is None:
                AVControl(speed=MIN_SPEED, angle=0)
                continue

            pid_angle = angle_pid.compute(0, -angle)  # target angle is 0 (straight line)
            print(f"Pid angle: {pid_angle:.2f} degrees")
            speed = calculate_speed(angle)  # Speed calculation remains unchanged
            
            AVControl(speed=speed, angle=pid_angle)

            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:        cv2.destroyAllWindows()  # Clean up any OpenCV windows
