from trainer.task import run

run('', True, 3, 'logs', ['isles_tfrecords'], num_epochs=10, learning_rate=1e-2)