# Author: Jose Sepulveda
# Description: This is a keras implementation of spectral normalization.
#              This was proposed in this paper: https://arxiv.org/pdf/1802.05957.pdf
#


from keras import backend as K


# Stochastic Gradient Descent with Spectral Normalization:
#   1) Initialize a random vector u, initialized from an isotropic distribution.
#   2) Use the Power iteration method with this vector u on the matrix of wieghts
#      to obtain two approximations of eigenvectors.
#   3) Calculate the spectral norm of the wieghts matrix.
#   4) Update wieghts using vanilla SGD using the spectral norm of the wieghts matrix.

def spectral_norm(w):
    """
        Input: tensor of wieghts
        Output: SN tensor of wieghts
    """
    def l2_norm(v):
        return K.sum(v ** 2) ** 0.5


    w_dim = w.shape.as_list()[-1]
    # Initialize random vector u
    u = K.random_normal(shape=[1, w_dim])

    # We need to flatten the wieghts
    w_flat = K.reshape(w, [-1, w_dim])

    # Power iteration method
    v = K.dot(u, K.transpose(w_flat))
    v = v / l2_norm(v)
    u = K.dot(v, w_flat)
    u = u / l2_norm(u)

    # Calculate the SN of W
    sigma = K.dot(K.dot(v, w_flat), K.transpose(u))
    w_sn = w_flat / sigma

    # Update wieghts
    w_sn = K.reshape(w_sn, w.shape.as_list())
    return w_sn

# test with random wieghts
W = K.random_normal((100, 100))
print(W)
spectral_norm(W)
