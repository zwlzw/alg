from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(r'D:\yolo-world\YOLO-World-master\yolov8s-world.pt')

    # model.set_classes(["person"])
    # model.set_classes(["car"])
    # model.set_classes(["dog"])
    model.set_classes(["person","car","dog"])

    video_path = r'D:\yolo-world\YOLO-World-master\data\video\1.mp4'
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():

        success, frame = cap.read()

        if success:

            results = model(frame)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO-World", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
