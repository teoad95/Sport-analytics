# Sport-analytics
Sport analytics using AI.

Develop an AI solution for sport analytics (football), which will:

- Detect players
  - on single frame
  - splitted window frame
- Classify players per team
  - using machine learning techniques
- Detect player's number in jersey (not done)
  - used pre-trained yolo model in SVHN dataset
- Image mapping into 2D football court (not done)
  - Generate dataset with frame - homography map
    - Generate database of homography maps, by zooming, tilting, etc.
  - Train GAN network to generate edge map images given frame input
  - Using HOG descriptor, find the best matching homography maps of database to the generated one
  - Plot image in 2D football court.
  

### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Tensorflow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html) or run it in [Google colab](https://colab.research.google.com/).

If you do not have Python installed yet, it is highly recommended to install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has some of the above packages and more included.

### Pre-trained networks - used tools

For:
- Players detection used [YOLO](https://pjreddie.com/darknet/yolo/) (You only look once) which is a real-time object detection system.
- Video mapping into 2D football court used [pix2pix CAN](https://github.com/mrzhu-cool/pix2pix-pytorch)

### Datasets

For:
- Video mapping into 2D football court, used [FIFA 2014 World Cup Dataset](http://www.cs.toronto.edu/~namdar/data/soccer_data.tar.gz)
- Detect player's number in jersey, used [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)

### References

[1] [Sports Video Analysis using Machine Learning](https://www.linkedin.com/pulse/sports-video-analysis-using-machine-learning-stephan-janssen/)

[2] [Automatic Birds Eye View Registration of Sports Videos](https://nihal111.github.io/hawk_eye/)

[3] [YOLO digit detector](https://github.com/penny4860/Yolo-digit-detector)

[4] [Pix2Pix Pytorch](https://github.com/mrzhu-cool/pix2pix-pytorch)
