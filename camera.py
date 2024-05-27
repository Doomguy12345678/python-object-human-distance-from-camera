import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Camera parameters (these are just placeholders)
FOV_horizontal_deg = 60
image_width_pixels = 640
known_width_ft = 50  # Width of object in ft
known_distance_ft = 100  # Distance at which the object has known width in ft

# Variables for motion tracking
prev_x = 0
prev_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Calculate area of contour
        area = cv2.contourArea(contour)
        
        # If area is greater than threshold, consider it as motion
        if area > 1000:
            # Draw bounding box around the object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate distance based on motion tracking
            distance_ft = known_distance_ft * known_width_ft / w
            
            # Display distance on the frame
            cv2.putText(frame, f"Distance: {distance_ft} ft", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
