# from ultralytics import YOLO
# import cv2

# # 1. Load YOLO model
# model = YOLO("best.pt")

# # 2. Load your image
# image = cv2.imread("down.jpg")

# # 3. Run YOLO on the image
# results = model(image)

# # 4. Draw boxes on the image
# annotated = results[0].plot()

# # 5. Show output
# cv2.imshow("YOLO Output", annotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2

# 1. Load YOLO model
model = YOLO("best.pt")   # small & fast model

# 2. Open video file or webcam
cap = cv2.VideoCapture("h3.mp4")   # put video path OR use 0 for webcam

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run YOLO detection
    results = model(frame, conf=0.2, iou=0.5)

    # 4. Draw results on frame
    annotated_frame = results[0].plot()

    # 5. Write frame to output video
    out.write(annotated_frame)

    # 6. Print detection results
    num_detections = len(results[0].boxes)
    print(f"Frame processed: {num_detections} helicopters detected")

    # 6. Optional: Show frame (comment out if running in headless environment)
    # cv2.imshow("YOLO Video Detection", annotated_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing complete. Output saved as 'output.mp4'")
