from numpy import array, zeros

kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])


def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the primal weight vector.
    """

    w = zeros(len(x[0]))
    # w = sum_i_to_m(alpha_i*x_i*y_i)
    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            w[j] += alpha[i]*y[i]*x[i][j]
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Given a primal support vector, return the indices for all of the support
    vectors
    """

    support = set()
    # For points on boundary (support vectors) wx + b = +-1
    for i in range(0, len(x)):
      if (abs(w.dot(x[i])+b - 1) <= tolerance or abs(w.dot(x[i])+b + 1) <= tolerance):
        support.add(i)
    return support


def find_slack(x, y, w, b):
    """
    Given a primal support vector instance, return the indices for all of the
    slack vectors
    """

    slack = set()
    # Slack vectors are those that have the wrong sign
    for i in range(0, len(x)):
      if y[i] == 1:
        if w.dot(x[i])+b <= 0:
          slack.add(i)
      else: # y[i] == -1
        if w.dot(x[i])+b >= 0:
          slack.add(i)
    return slack
