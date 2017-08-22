from trainer.task import run

run('', True, 1, 'logs', ['training_tfrecords'], num_epochs=10, learning_rate=1e-2)