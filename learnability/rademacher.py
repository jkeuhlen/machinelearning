from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, radians

from numpy import array
from numpy.linalg import norm

from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


class Classifier:
    def correlation(self, data, labels):
        """
        Return the correlation between a label assignment and the predictions of
        the classifier

        Args:
          data: A list of datapoints
          labels: The list of labels we correlate against (+1 / -1)
        """

        assert len(data) == len(labels), \
            "Data and labels must be the same size %i vs %i" % \
            (len(data), len(labels))

        assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

        # Correlation is 1/m * sum_i=1_to_m(y_i*h(x_i))

        m = len(data)
        sum = 0.0
        for i in range(0,m):
            #Convert hypothesis to +/- 1 from bool
            h = self.classify(data[i])
            if (h == True):
                h = 1
            else:
                h = -1
            sum += labels[i]*h
        cor = sum/m #Notes have 1/2m? Double check this

        return cor


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector

        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        #print "Vector: ", self._vector
        #print "Point: ", point
        #print "Dot Product: ", self._vector.dot(point)
        #print "Class: ", self._vector.dot(point) >= 0
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """
    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.

        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.

    """
    def __init__(self, start_x, start_y, end_x, end_y):
        """

        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)

        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point

        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.

    Args:
      dataset: The dataset to use to generate hypotheses

    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # Our hyperplanes are really just lines in two dimensional space.
    # Since each goes through the origin, just try the vector that reaches each point in the dataset. Use a vector from each point and calculate its slope from the origin then define our lines using y=mx+b (b=0 always)
    # Skip points on the same line
    # Use the normal vectors (each forces classification in one direction)
    # In order to get unique classifications, we need to shift each decision boundary slightly. As in the picture, I'll make each line slightly steeper than the vector to points.
    a = []
    for vec in dataset:
        x = vec[0]
        y = vec[1]
        slope = y/x
        # Don't yield lines that have a slope already used
        if slope in a:
            continue
        # Rotate by 1 degree to get normal vector of slightly steeper line!
        x = x*cos(radians(1))-y*sin(radians(1))
        y = x*sin(radians(1))+y*cos(radians(1))
        # We need to specify the normal vector to the decision boundary
        #  So flip the order and sign 
        # One sign change for each direction (+/- = left/right)
        yield OriginPlaneHypothesis(-y, x)
        yield OriginPlaneHypothesis(y,-x)
        a.append(slope)

def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.

    Args:
      dataset: The dataset to use to generate hypotheses

    """

    # Complete this for extra credit
    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.

    Classifiers are axis-aligned rectangles

    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # Based on VC-Dimension, we can only ever have 4 points
    # Inside rectangles is positive
    # 0 Positive - take x_min-1, y_min-1 make that a rectangle-point
    # 1 Positive - each point is its own rectangle
    # 2 Positive - each pair of points
    # 3 Positive - much be contained within the extreme pairs
    #  Need to make sure I'm not double-classifying somehow
    # 4 Positive - x_min-x_max, y_min-y_max
    
    # Min/Max variables
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    # Create a BST and fill it with all of the points in the dataset
    bst = BST()
    map(bst.insert, dataset)
    # list to hold pair combos
    a = []

    # Dataset is a list of tuples of points
    for start_point in dataset:
        start_x = start_point[0]
        start_y = start_point[1]
        yield AxisAlignedRectangle(start_x, start_y, start_x, start_y)
        # Calculate min/maxes
        if start_x > x_max:
            x_max = start_x
        elif start_x < x_min:
            x_min = start_x
        if start_y > y_max:
            y_max = start_y
        elif start_y < y_min:
            y_min = start_y       
        for end_point in dataset:
            points_in_range = []
            end_x = end_point[0]
            end_y = end_point[1]
            # Manipulate points that only cause half negative
            if (end_x > start_x and end_y < start_y):
                start_y = end_y
            elif (end_x < start_x and end_y > start_y):
                start_x = end_x
            # Skip points that cause fully negative rectangles
            if (end_x <= start_x and end_y <= start_y):
                continue
            points_in_range = map(lambda x: x.key, list(bst.range(start_point, end_point)))
            if points_in_range in a:
                continue
            else:
                yield AxisAlignedRectangle(start_x, start_y, end_x, end_y)
                a.append(points_in_range)
    # Get the 4 point cases
    # All in
    yield AxisAlignedRectangle(x_min-1, y_min-1, x_max+1, y_max+1)
    # All out
    yield AxisAlignedRectangle(x_min-1, y_min-1,x_min-1, y_min-1)


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.

    Args:
      number: The number of coin tosses to perform

      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in xrange(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity

    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator

      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset

      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """

    # TODO: complete this function
    #for ii in xrange(num_samples):
    #    if random_seed != 0:
    #        rademacher = coin_tosses(len(dataset), random_seed + ii)
    #    else:
    #        rademacher = coin_tosses(len(dataset))
    # R(H) = E_sig[max_h_in_H(1/m*sum_i_m(sig_i*h(x_i)))]
    # Do this whole thing num_samples times to get Expectation value
    m = len(dataset)
    expecation_final = 0.0
    for i in range(0,num_samples):
        if random_seed != 0:
            rademacher = coin_tosses(len(dataset), random_seed + i)
        else:
            rademacher = coin_tosses(len(dataset))
        array = []
        hyps = hypothesis_generator(dataset)
        for h in hyps:
            sum = 0.0
            for i in range(0,m):
                x = h.classify(dataset[i])
                #Convert hypothesis to +/- 1 from bool
                if (x == True):
                    x = 1
                else:
                    x = -1
                sum += rademacher[i]*x
            rad = sum/m
            array.append(rad)
        final = max(array)
        expecation_final += final
    expecation_final = expecation_final/num_samples
    return expecation_final

if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))
    print("Rademacher correlation of plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))

