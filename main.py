from ultralytics import YOLO, settings
import cv2
import glob
import os
import matplotlib.pyplot as plt

def main():
    # # Atualiza a pasta onde os resultados serão salvos
    # settings.update({'runs_dir': './runs'})

    # # Caminhos
    # DATASET_PATH = "C:/Users/Junão/Desktop/projeto_yolo_carros/projeto_yolo_carros/datasets/vehicles.v2-release.yolov8/data.yaml"
    IMAGE_PATH = r'C:\Users\Junão\Desktop\projeto_yolo_carros\projeto_yolo_carros\datasets\vehicles.v2-release.yolov8\test\images\adit_mp4-5_jpg.rf.bd945716e20cb3f850e2ad36df03d6e3.jpg'

    # # Treinamento
    # modelo = YOLO('yolov8s.pt')
    # modelo.train(
    #     data=DATASET_PATH,
    #     epochs=100,
    #     batch=32,
    #     imgsz=640,
    #     patience=8,
    #     pretrained=True,
    #     val=True,
    #     workers=6,  
    #     device='cuda',
    #     single_cls=False,
    #     box=7.5,
    #     cls=0.5,
    #     dfl=1.5,
    #     degrees=0.3,
    #     hsv_s=0.3,
    #     hsv_v=0.3,
    #     scale=0.5,
    #     fliplr=0.5
    # )

    # Validação
    modelo = YOLO('runs/detect/train3/weights/best.pt')  # ajuste o caminho se necessário
    modelo.val(
        imgsz=640,
        batch=16,
        conf=0.001,
        iou=0.7,
        save_json=False,
        split='test'
    )

    # Predição
    results_img = modelo.predict(
        source=IMAGE_PATH,
        conf=0.35,
        iou=0.7,
        imgsz=640,
        save=True,
        save_txt=True,
        save_conf=True,
        save_crop=True
    )

    # Procura arquivos .jpg na pasta de saída
    predict_dir = 'runs/detect/predict'
    jpg_files = glob.glob(os.path.join(predict_dir, '*.jpg'))

    # Exibe a imagem se encontrada
    if jpg_files:
        result_path = jpg_files[0]  # Pega o primeiro arquivo
        img = cv2.imread(result_path)
        if img is None:
            print(f"Erro ao carregar imagem: {result_path}")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title('Detecções YOLOv8')
        plt.show()
    else:
        print("Nenhuma imagem encontrada em runs/detect/predict")

if __name__ == "__main__":
    main()
