import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def m():
    flags.DEFINE_string('name', None,
                        'Append a name Tag to run.')

    flags.DEFINE_string('hypes', 'hypes/medseg.json',
                        'File storing model parameters.')


if __name__ == '__main__':
    tf.app.run(m)
