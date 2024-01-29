from os import walk

classes_in = ['tank', 'apc', 'helicopters', 'desert']
classes = ['tank', 'apc', 'helicopters']

data_folder = r'data/'
num_classes = len(classes)
image_shape = 100
save_every = 1000
batch_size = 256
epochs = 90
file_count = sum(len(files) for _, _, files in walk(data_folder))
validation_split = 0.3
training_files_count = file_count * (1 - validation_split)
num_of_batches = int(epochs * training_files_count / batch_size) + 19
epoch_size = int(training_files_count / batch_size)
save_every = min(save_every, epoch_size)
show_last_x_batches = max(save_every, int(num_of_batches * .05))
learning_rate = .5e-6
