from trainer.task import run
run('', True, 10, 'logs', ['isles2017.tfrecord'], num_epochs=10)