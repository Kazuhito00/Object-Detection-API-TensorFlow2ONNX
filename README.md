# Object-Detection-API-TensorFlow2ONNX

[TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)のONNX推論サンプルです。<br>
ONNXに変換したモデルも同梱しています。<br>
変換自体を試したい方は[Object-Detection-API-TensorFlow2ONNX.ipynb](Object-Detection-API-TensorFlow2ONNX.ipynb)をColaboratoryで実行してください。<br>

https://user-images.githubusercontent.com/37477845/153884020-28278d48-7792-4d7c-9be7-7fba82b548b0.mp4

# Requirement 
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later

# Demo
デモの実行方法は以下です。
```bash
python demo.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/efficientdet_d0.onnx
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：512,512
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.5

# Reference
* [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
Object-Detection-API-TensorFlow2ONNX is under [Apache-2.0 License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[イギリス ウースターのエルガー像](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002011239_00000)を使用しています。
