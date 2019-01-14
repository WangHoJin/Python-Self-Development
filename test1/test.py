"""
    TEST2
    ~~~~~
"""
import numpy as np

# 신경망 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity(x):
    """

    :param x:
    :return:
    """
    return x

def init_network():
    """

    :return:
    """

    network = {}
    network['W1'] = np.array([[1, 3, 5], [2, 4, 6]])  # weight 은 관습적으로 W라고 합니다.
    network['W2'] = np.array([[1, 2], [3, 4], [5, 6]])
    network['W3'] = np.array([[1, 2], [3, 4]])
    return network


# 신경망 구현
def forward(network, x):
    """미리 입력받은 a와 b값이 같은지 확인하여 결과를 반환합니다.

           :return: boolean True or False에 대한 결과, a와 b가 값으면 True, 다르면 False

           예제:
               다음과 같이 사용하세요:

               >>> Test(1, 2).is_same()
               False

           """
    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    y = np.dot(x, W1)
    y_hat = sigmoid(y)
    k = np.dot(y_hat, W2)
    k_hat = sigmoid(k)
    j = np.dot(k_hat, W3)
    j_hat = identity(j)
    """identity 대신 softmax함수를 사용할 수 있습니다."""
    return j_hat

network = init_network()

x = np.array([1, 2])


"""입력"""
y = forward(network, x)  # 출력
print(y)
