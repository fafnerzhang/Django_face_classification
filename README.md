# Django_face_classification
Google相簿有一功能是可將有相同人臉的照片分在同一類，因為對相關技術與如何實作有興趣，便進行相關技術的研究，傳統的人臉分類模型最後輸出層為softmax，只要多加一張人臉便必須重新訓練整個網路，網路上搜尋可解決此問題的技術後，決定使用Facenet與Django進行實作

### FaceNet
![](https://i.imgur.com/uQFFjfs.png)

FaceNet的網路輸入一批量資料後會經由類神經網路(此專案中為InceptionResnetV1),然後進行L2正規化,得到人臉嵌入,最後利用三元組損失進行反向傳播。使用深度卷積網路學習每個圖像的歐基里德嵌入(Euclidean embedding)，網路經過訓練使人臉映射到一嵌入空間，同一人的面部特徵會有較小的距離且不同人的面部會有較大距離，因此我們可透過設定閥值來判斷是否為同一人。

本次專案中使用pytorch版本的facenet做為主要辨識網路進行實作。　
https://github.com/timesler/facenet-pytorch

## 專案架構
![](https://i.imgur.com/9gEfgun.png)

