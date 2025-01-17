# 3.1 MNIST
# 손글씨 이미지의 MNIST 데이터셋을 사용, 머신러닝 분야의 Hello, worlds 라고 볼 수 있다.

from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784",version=1, as_frame=False)
key = mnist.keys()
print(key)
