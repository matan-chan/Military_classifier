import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from config import num_of_batches, validation_split, show_last_x_batches, classes, learning_rate, epoch_size
from data import prepare_dataset, clean_last_train, preprocess_images
from model import costume_model
from keras.metrics import CategoricalAccuracy, Precision, Recall
from contextlib import redirect_stdout
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from tensorflow import reduce_sum
from config import save_every
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from io import StringIO
import numpy as np
import shutil
import pytz
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        print(e)


class MilitaryClassifier:

    def __init__(self):
        self.model = costume_model()
        self.bce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.optimizer = Adam(learning_rate=learning_rate)
        self.accuracy_metric = CategoricalAccuracy()
        self.precision_metric = Precision()
        self.recall_metric = Recall()
        self.metrics = {}
        self.update_metrics_history(0)
        self.model_id = ''
        self.manager = self.manage_checkpoints()

    def train(self):
        self.initialize_model_id()
        dataset = prepare_dataset()
        start_batch = self.get_starting_batch()
        bar = self.get_bar(start_batch)

        for batch_number, (batch, labels) in enumerate(dataset, start_batch):
            loss = self.train_step(batch, labels)
            bar.update(batch_number + start_batch, values=self.get_formatted_metrics(loss))
            if (1 + batch_number) % save_every == 0:
                self.plot_metrics(start_batch, batch_number + 1, loss)
                self.manager.save()

    def train_step(self, batch, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(batch)
            loss = self.bce(labels, predictions)
            self.update_metrics(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_weights))

        return reduce_sum(loss).numpy()

    def plot_metrics(self, starting_batch: int, batch_number: int, loss: float):
        self.update_metrics_history(loss)
        end_plot_batch = batch_number + save_every
        batches = range(starting_batch, end_plot_batch, save_every)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
        for key in self.metrics.keys():
            self.plot_graph(ax1, batches, self.metrics[key], batch_number, key)
            if key != 'loss':
                g2_range, last_x_batches_metrics = self.calc_g2_range(self.metrics[key], batch_number)
                self.plot_graph(ax2, g2_range, last_x_batches_metrics, batch_number, key)
                ax2.text(g2_range.start, last_x_batches_metrics[0], f'{last_x_batches_metrics[0]}', ha='left',
                         va='bottom', fontsize=8, color='red')
                self.plot_graph(ax3, batches, self.metrics[key], batch_number, key)

        self.add_info_to_graph(ax1, f'Training Metrics batch {batch_number}')
        self.add_info_to_graph(ax2, f'Training Metrics last {show_last_x_batches} batches. batch {batch_number}')
        self.add_info_to_graph(ax3, f'Training Metrics batch {batch_number} without loss')
        self.handel_graphs_file(batch_number)

    def update_metrics_history(self, loss: float):
        new_metrics = self.get_metrics(loss)
        if self.metrics:
            for key in self.metrics.keys():
                self.metrics[key].append(new_metrics[key])
        else:
            for key in new_metrics:
                self.metrics[key] = [new_metrics[key]]

    def get_formatted_metrics(self, loss: float):
        metrics = self.get_metrics(loss)
        formatted_metrics = [
            ("accuracy", metrics['accuracy']),
            ('loss', metrics['loss']),
            ("precision", metrics['precision']),
            ("recall", metrics['recall']),
            ("f1_score", metrics['f1_score'])
        ]
        return formatted_metrics

    def get_metrics(self, loss: float):
        metrics = {'precision': self.precision_metric.result().numpy(),
                   'recall': self.recall_metric.result().numpy(),
                   'accuracy': self.accuracy_metric.result().numpy(),
                   'f1_score': 0.0,
                   'loss': loss}

        if metrics['precision'] and metrics['recall']:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                    metrics['precision'] + metrics['recall'])

        return metrics

    def handel_graphs_file(self, batch_number: int):
        epoch = int(batch_number / epoch_size)
        plt.savefig(f'logs/{self.model_id}/{self.model_id} epoch {epoch}', dpi=300)
        plt.clf()
        plt.close()

    @staticmethod
    def plot_graph(graph, batches: range, data_points: [float], batch_number: int, label: str):
        graph.plot(batches, data_points, label=label)
        graph.text(batch_number, data_points[-1], f'{data_points[-1]}', ha='right', va='bottom', fontsize=8,
                   color='red')

    @staticmethod
    def add_info_to_graph(graph, title: str):
        graph.grid(True, linestyle='-', alpha=0.7)
        graph.set_title(title)
        graph.set_xlabel('batch')
        graph.legend()

    @staticmethod
    def calc_g2_range(data_points: [float], batch_number: int):
        end_plot_batch = batch_number + save_every
        last_x_batches_metrics = data_points[-int(show_last_x_batches / save_every) - 1:]
        start_plot_batch = batch_number - save_every * (len(last_x_batches_metrics) - 1)
        return range(start_plot_batch, end_plot_batch, save_every), last_x_batches_metrics

    def update_metrics(self, labels_confusion_matrix, distance_confusion_matrix):
        self.accuracy_metric.update_state(labels_confusion_matrix, distance_confusion_matrix)
        self.precision_metric.update_state(labels_confusion_matrix, distance_confusion_matrix)
        self.recall_metric.update_state(labels_confusion_matrix, distance_confusion_matrix)

    def evaluate_model(self):
        dataset = prepare_dataset('validation')
        total_loss = 0.0
        num_batches = num_of_batches * validation_split
        for images, labels in dataset:
            loss, predictions = self.custom_evaluation_step(images, labels)
            total_loss += loss.numpy()
        average_loss = total_loss / num_batches
        metrics_summary = self.get_metrics(average_loss)
        print(metrics_summary)
        if self.model_id != '':
            with open(f'logs/{self.model_id}/{self.model_id}.txt', 'a') as file:
                file.write(str(metrics_summary))
        ######
        if self.accuracy_metric.result().numpy() > 0.8 and self.precision_metric.result().numpy() > 0.8:
            source_directory = 'models/'
            destination_directory = f'logs/{self.model_id}/'
            files_to_copy = os.listdir(source_directory)
            for file_name in files_to_copy:
                source_path = os.path.join(source_directory, file_name)
                destination_path = os.path.join(destination_directory, file_name)
                shutil.copy(source_path, destination_path)
        #########
        else:
            shutil.rmtree(f'logs/{self.model_id}/')

    def custom_evaluation_step(self, images, labels):
        predictions = self.model(images, training=False)
        loss = self.bce(labels, predictions)
        self.update_metrics(labels, predictions)
        return loss, predictions

    def manage_checkpoints(self):
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(checkpoint, directory=f'models/', max_to_keep=1)
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        return manager

    def get_bar(self, start_batch: int):
        if self.manager.latest_checkpoint:
            bar = tf.keras.utils.Progbar(num_of_batches * 2 + start_batch)
        else:
            bar = tf.keras.utils.Progbar(num_of_batches)
        return bar

    def get_starting_batch(self) -> int:
        if self.manager.latest_checkpoint:
            start_batch = int(self.manager.latest_checkpoint.split(sep='ckpt-')[-1]) * save_every
        else:
            self.create_model_log()
            start_batch = 0
        return start_batch

    def initialize_model_id(self):
        if self.manager.latest_checkpoint:
            datetime_object = datetime.utcfromtimestamp(self.get_model_timestamp())
            israel_tz = pytz.timezone('Asia/Jerusalem')
            datetime_object_israel = datetime_object.replace(tzinfo=pytz.utc).astimezone(israel_tz)
            self.model_id = datetime_object_israel.strftime("%Y-%m-%d %H-%M")
        else:
            self.model_id = datetime.now().strftime("%Y-%m-%d %H-%M")

    @staticmethod
    def get_model_timestamp() -> float:
        with open('models/checkpoint', 'r') as src_file:
            file_content = src_file.read()
            lines = file_content.split('\n')
            last_preserved_timestamp = float(lines[-2].split(':')[-1].strip())
        return last_preserved_timestamp

    def create_model_log(self):
        Path(f'logs/{self.model_id}').mkdir(parents=True)
        with open('config.py', 'r') as src_file:
            file_content = src_file.read()
            summary_buffer = StringIO()
            with redirect_stdout(summary_buffer):
                self.model.summary()
            summary_string = summary_buffer.getvalue()

            with open(f'logs/{self.model_id}/{self.model_id}.txt', 'w') as file:
                file.write(file_content + '\n' + summary_string)

    def classify(self, images):
        results = self.model(images, training=False)
        row_contains_higher_than_thresh_hold = np.any(results > 0.4, axis=1)
        encodes = np.argmax(results, axis=1)
        encodes[~row_contains_higher_than_thresh_hold] = -1
        return encodes

    @staticmethod
    def move_to(images, encodes, name):
        for i, encode in enumerate(encodes):
            if encode != -1:
                des_dir = classes[encode]
            else:
                des_dir = 'none'
            cv2.imwrite(f'output/{des_dir}/{name}', images[i])

    def predict(self):
        clean_last_train(['output/tank', 'output/apc', 'output/helicopters', 'output/desert'])
        directory_path = f'test'
        correct = 0
        sum = 0
        for d in os.listdir(directory_path):
            for f in os.listdir(f'{directory_path}/{d}'):
                try:
                    image = cv2.imread(f'{directory_path}/{d}/{f}')
                    images = np.array([image])
                    pi = preprocess_images(images)
                    pi = tf.reverse(pi, axis=[-1])
                    encodes = self.classify(pi)
                    sum += 1
                    if classes[encodes[0]] == d:
                        correct += 1
                    self.move_to(images, encodes, f)
                except:
                    print('fail')
        print(correct / sum * 100)
