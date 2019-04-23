from fastai.callbacks import *
from fastai.vision import *
from fastai.vision.learner import *
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model)
learn.metrics = [accuracy]
learn.fit(1)
# learn.recorder.plot_lr(show_moms=True)
cb = OneCycleScheduler(learn, lr_max=0.01)
learn.fit(1, callbacks=cb)
learn.fit_one_cycle(1)
learn = cnn_learner(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1)
