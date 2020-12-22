import tensorflow.compat.v1 as tf
from keras.callbacks import TensorBoard

tf.disable_v2_behavior()

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        self._write_logs(stats, step)
