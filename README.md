# tensorflow-practice

## 環境構築
- Nvidia GPU Cloud に登録後、以下を実行。
```
docker run --gpus all -it --rm -v /home/sato/work/tensorflow-practice:/work/tensorflow-practice nvcr.io/nvidia/tensorflow:20.01-tf2-py3 /bin/bash
```

## 内容
- mnist
    - mnist データを簡単なネットワークを作って学習・評価する一連の処理を記載