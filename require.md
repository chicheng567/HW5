1. 幫我把整個訓練改成yolo v3，但是你只能修改yolo_loss.py 和src/resnet_yolo.py
2. Backbone使用torchvision的denseNet，此項修改只能改在src/resnet_yolo.py
3. 希望你在改寫src/resnet_yolo.py要模組化，分為:Backbone, predictionHead(neck也包含在這部分)。最後再用一個ODmodel class封裝。如果有需要NMS也寫在這個檔案，且只有在inference mode的時候使用。
4. yolo_loss.py 也需要模組化，分為幾個part：preprocessing part(將gt轉換成能跟pred算loss的形式)、loss part
5. 請使用/miniconda/envs/vl3/bin/python這個虛擬環境。