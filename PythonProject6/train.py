from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    # Load a model
    multiprocessing.freeze_support()

    model = YOLO("best.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="yolo.yaml", epochs=100, imgsz=640, batch=16, device="cuda")