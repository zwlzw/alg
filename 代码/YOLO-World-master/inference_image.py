from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\yolo-world\YOLO-World-master\yolov8s-world.pt')

    # model.set_classes(["person"])
    # model.set_classes(["car"])
    # model.set_classes(["dog"])
    model.set_classes(["person","car","dog"])

    results = model.predict(r'D:\yolo-world\YOLO-World-master\data\img\1.jpg')
    # results = model.predict(r'D:\yolo-world\YOLO-World-master\data\img\2.jpg')

    results[0].show()


