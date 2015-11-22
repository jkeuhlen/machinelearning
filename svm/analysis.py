from sklearn import svm

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


data = Numbers("../data/mnist.pkl.gz")
x = []
y = []

for i in range(0, len(data.train_y)):
    if (data.train_y[i] == 3 or data.train_y[i] == 8):
        y.append(data.train_y[i])
        x.append(data.train_x[i])

classifier = svm.SVC(C=0.1, kernel='linear')
classified = classifier.fit(x, y)

test_x = []
test_y = []

for i in range(0, len(data.test_y)):
    if (data.test_y[i] == 3 or data.test_y[i] == 8):
        test_y.append(data.test_y[i])
        test_x.append(data.test_x[i])
# Accuracy
count = len(test_y)
accuracy = 0
for i in range(0, count):
    if (classified.predict(test_x[i]) == test_y[i]):
        accuracy += 1
print "Accuracy: ", float(accuracy)/count*100.0

# Picture
svs = classifier.support_vectors_
numpix = 28
bp = [numpix*i for i in range(numpix+1)]
image = [x[classified.support_[len(classified.support_)-1]][bp[i]:bp[i+1]] for i in range(numpix)]
show(image)

