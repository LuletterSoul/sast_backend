import tensorflow as tf
from tensorboardX import SummaryWriter


# class Logger(object):
#     """Tensorboard logger."""
#
#     def __init__(self, log_dir):
#         """Initialize summary writer."""
#         self.writer = tf.compat.v1.summary.FileWriter(log_dir)
#
#     def scalar_summary(self, tag, value, step):
#         """Add scalar summary."""
#         summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

    def scalars_summary(self, tag, values, step):
        self.writer.add_scalars(tag, values, step)
