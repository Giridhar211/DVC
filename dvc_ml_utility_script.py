from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

results = model.train(data=r'C:\Users\Giridhar\PycharmProjects\DVC\dataset\data.yaml',
                      epochs=3)

results = model.val()

results = model('https://ultralytics.com/images/bus.jpg')

success = model.export(format='onnx')