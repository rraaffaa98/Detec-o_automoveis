Este projeto tem como objetivo detectar veículos em imagens utilizando o modelo YOLOv8.
A proposta é aplicar visão computacional e aprendizado profundo para identificar diferentes tipos de veículos em imagens reais.
O dataset utilizado foi obtido gratuitamente no site do Roboflow, mais especificamente no seguinte link: https://universe.roboflow.com/roboflow-100/vehicles-q0x2v. 
Esse conjunto de dados contém diversas classes de veículos, incluindo carros, ônibus e caminhões de diferentes tamanhos.

O modelo escolhido para o projeto foi o YOLOv8s, pois ele representa um equilíbrio entre o YOLOv8n (modelo menor e mais rápido de treinar) e o YOLOv8x (modelo maior e mais robusto, porém inviável para a máquina utilizada). 
Algumas técnicas de aumento de dados foram aplicadas, como modificações de brilho, rotação, escala e espelhamento horizontal. 
Além disso, foram realizados ajustes em parâmetros como o número de épocas, tamanho do batch, thresholds de confiança e interseção sobre união (IoU).

É importante destacar que o projeto foi desenvolvido em um notebook com processador Intel de oitava geração, 8 GB de RAM e uma GPU limitada. 
Por conta disso, algumas escolhas foram feitas considerando as limitações computacionais da máquina. 
Por exemplo, não foi possível utilizar o modelo YOLOv8x nem explorar de forma mais intensiva as técnicas de data augmentation, para não comprometer o desempenho da máquina durante o treinamento.
