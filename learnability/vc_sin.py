from math import pi, sin

kSIMPLE_TRAIN = [(1, False), (2, True), (4, False), (5, True), (13, False),
                 (14, True), (19, False)]


class SinClassifier:
    """
    A binary classifier that is parameterized by a frequency.
    """

    def __init__(self, frequency):
        """
        Create a new classifier parameterized by frequency \omega

        Args:
          frequency: The frequency of the sin function (a real number)
        """
        assert isinstance(frequency, float)
        self._frequency = frequency

    def __call__(self, x):
        """
        Returns the raw output of the classifier.  The sign of this value is the
        final prediction.

        Args:
          x: The data point (an integer)
        """
        return sin(self._frequency * 2 ** (-x))

    def classify(self, x):
        """

        Classifies an integer based on whether the sign of \sin(\omega * 2^{-x})
        is >= 0.  If it is, the classifier returns True.  Otherwise, false.

        Args:
          x: The data point (an integer)
        """
        assert isinstance(x, int), "Object to be classified must be an integer"

        if self(x) >= 0:
            return True
        else:
            return False


def train_sin_classifier(data):
    """
    Compute the correct frequency of a classifier to prefectly classify the
    data and return the corresponding classifier object

    Args:
      data: A list of tuples; first coordinate is x (integers), second is y (+1/-1)
    """

    assert all(isinstance(x[0], int) and x >= 0 for x in data), \
        "All training points must be integers"
    assert all(isinstance(x[1], bool) for x in data), \
        "All labels must be True / False"

    # TODO: Compute a frequency that will correctly classify the dataset
    frequency = 1.0
    for tup in data:
        x = tup[0]
        y = tup[1]
        #Convert hypothesis to +/- 1 from bool
        if (y == True):
            y = 1
        else:
            y = -1
        # Solve sin(w*x) = y for the set of equations to follow
        # w = (1/x)*Arcsin(y)
        # Arcsin(+1) = +pi/2
        # Arcsin(-1) = -pi/2
        frequency = 2**x*y*pi/2 
    return SinClassifier(frequency*pi)

if __name__ == "__main__":
    classifier = train_sin_classifier(kSIMPLE_TRAIN)
    for xx, yy in kSIMPLE_TRAIN:
        print(xx, yy, classifier(xx), classifier.classify(xx))
