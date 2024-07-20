import cv2
import numpy as np
import time

coins = {
    1: {"name": "2", "radius": None, "color": None},
    2: {"name": "1", "radius": None, "color": None},
    3: {"name": "0.50", "radius": None, "color": None},
    4: {"name": "0.20", "radius": None, "color": None},
    5: {"name": "0.10", "radius": None, "color": None},
    6: {"name": "0.05", "radius": None, "color": None}
}

alert_text = ""
alert_start_time = 0
alert_duration = 0

show_all_circles = True

def classify_coin(radius, mean_color):
    closest_coin = 0
    min_distance = float('inf')
    threshold_distance = 35

    for key, properties in coins.items():
        # Check if this coin has properties saved yet
        if properties["radius"] is None or properties["color"] is None:
            continue
        
        # Calculate distance for radius and color
        radius_distance = abs(radius - properties["radius"])
        color_distance = np.linalg.norm(np.array(mean_color) - np.array(properties["color"]))
        
        # Combine distances
        distance = radius_distance + color_distance * 0.5
        
        # Update closest coin if new closest found
        if distance < threshold_distance and distance < min_distance:
            min_distance = distance
            closest_coin = key
    
    return closest_coin, min_distance

def detect_and_draw_circles(frame):
    global show_all_circles

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=0.5, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=100)

    # Store the keys of the detected coins
    circle_keys = []
    
    # Draw circles if any
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

            # Create a mask for the circle
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), r, 255, -1)
            
            # Calculate the mean color inside the circle
            mean_val = cv2.mean(frame, mask=mask)[:3]
            mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2])) # BGR

            coin_dist = float('inf')
            key = [0,0,0,0,0,0,0,0,0,0]
            # Classify the coin
            for i in range(10):
                key[i], coin_dist = classify_coin(r, mean_color)

            coin_key = most_frequent(key)

            # Save the key of the detected coin
            if coin_key != 0:
                circle_keys.append(coin_key)
                cv2.putText(frame, f"{coins[coin_key]['name']}", (x - 25, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if coin_dist < float('inf'):
                cv2.putText(frame, f"{coin_dist:.2f}", (x - 25, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            #cv2.putText(frame, f"{r:.2f}", (x - 25, y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            #cv2.putText(frame, f"{mean_color}", (x - 50, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
    return frame, circles, circle_keys

def calculate_sum(keys):
    total_sum = 0
    for key in keys:
        total_sum += float(coins[key]["name"])
    return total_sum

def set_alert(text, duration=2):
    global alert_text, alert_start_time, alert_duration
    alert_text = text
    alert_start_time = time.time()
    alert_duration = duration

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def main():
    global alert_text, alert_start_time, alert_duration, show_all_circles

    sum_buffer = [0,0,0,0,0,0,0,0,0,0]

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame.")
            break
        
        # Detect and draw circles
        output_frame, circles, keys = detect_and_draw_circles(frame)

        sum = calculate_sum(keys)
        sum_buffer.pop(0)
        sum_buffer.append(sum)

        # Draw the total sum
        cv2.putText(output_frame, f"Total: {most_frequent(sum_buffer):.2f}e", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        
        # Draw alert text
        if alert_text != "":
            cv2.putText(output_frame, alert_text, (10, int(cap_height) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
            if time.time() - alert_start_time > alert_duration:
                alert_text = ""

        # Display the frame
        cv2.imshow('Coin Detection', output_frame)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite('detected_circles.png', output_frame) # Save frame as image
        elif key == ord('q'):
            break # Quit
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
            coin_index = key - ord('0')

            if coin_index in coins:
                coin_properties = coins[coin_index]

                if circles is not None:
                    for (x, y, r) in circles:
                        # Create a mask for the circle
                        mask = np.zeros(frame.shape[:2], dtype="uint8")
                        cv2.circle(mask, (x, y), r, 255, -1)
                        
                        # Calculate the mean color inside the circle
                        mean_val = cv2.mean(frame, mask=mask)[:3]
                        mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2])) # BGR order
                        
                        # Update coin properties
                        coin_properties["radius"] = r
                        coin_properties["color"] = mean_color

                        # Show alert
                        set_alert(f"Saved {coin_properties['name']}e")

                        break
    
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
