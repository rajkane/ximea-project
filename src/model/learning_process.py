import os
import torchvision
import torch
from src.external import qtc
from detecto import utils
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .rcnn_model import Dataset, DataLoader
from tqdm import tqdm


class WorkerRCNN(qtc.QThread):
    """
    deep learning process
    """
    learn = qtc.pyqtSignal(str, name="deep-learning-process")
    learn_graph = qtc.pyqtSignal(list)
    status = qtc.pyqtSignal(str, name="bar-status")
    exception = qtc.pyqtSignal(str, name="exception")
    enabled_learning_process = qtc.pyqtSignal(bool, name="enabled/disabled-GUI-objects")

    def __init__(self, dataset_name: str, batch_size: int, annotation: list, epochs: int,
                 lr_step_size: int, learning_rate: float, model_name: str):
        super(WorkerRCNN, self).__init__()
        self.thread_active = False
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.annotation = annotation
        self.epochs = epochs
        self.lr_step_size = lr_step_size
        self.learning_rate = learning_rate
        self.resize = None
        self.horizontal = None
        self.vertical = None
        self.autocontrast = None
        self.equalize = None
        self.rotation = None
        self.model_name = model_name
        self.settings = None
        self.custom_transforms = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()

        self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        if self.annotation:
            # Get the number of input features for the classifier
            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            # Replace the pre-trained head with a new one (note: +1 because of the __background__ class)
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.annotation) + 1)
            self._disable_normalize = False

        self._model.to(self.device)

        # Mappings to convert from string labels to ints and vice versa
        self._classes = ['__background__'] + self.annotation
        self._int_mapping = {label: index for index, label in enumerate(self._classes)}

    def status_interrupt(self):
        self.learn.emit("\tDeep Learning has been interrupted!")
        self.exception.emit("Deep Learning has been interrupted!")

    # Converts all string labels in a list of target dicts to
    # their corresponding int mappings
    def convert_to_int_labels(self, targets):
        for idx, target in enumerate(targets):
            # get all string labels for objects in a single image
            labels_array = target['labels']
            # convert string labels into one hot encoding
            labels_int_array = [self._int_mapping[class_name] for class_name in labels_array]
            target['labels'] = torch.tensor(labels_int_array)

    # Sends all images and targets to the same device as the model
    def to_device(self, images, targets):
        images = [image.to(self.device) for image in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def save(self, file):
        torch.save(self._model.state_dict(), file)

    def set_resize(self, resize):
        self.resize = resize

    def set_random_horizontal_flip(self, horizontal):
        self.horizontal = horizontal

    def set_random_vertical_flip(self, vertical):
        self.vertical = vertical

    def set_random_autocontrast(self, autocontrast):
        self.autocontrast = autocontrast

    def set_random_equalize(self, equalize):
        self.equalize = equalize

    def set_random_rotation(self, rotation):
        self.rotation = rotation

    def get_augment(self):
        """self.settings = qtc.QSettings("augmentation.ini", qtc.QSettings.format.IniFormat)
        resize = int(self.settings.value("resize", int))
        r_horizontal_flip = float(self.settings.value("random-horizontal-flip", float))
        r_vertical_flip = float(self.settings.value("random-vertical-flip", float))
        r_auto_contrast = float(self.settings.value("random-auto-contrast", float))
        r_equalize = float(self.settings.value("random-equalize", float))
        r_rotation_min = int(self.settings.value("random-rotation-min", int))
        r_rotation_max = int(self.settings.value("random-rotation-max", int))"""

        self.custom_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomHorizontalFlip(p=self.horizontal),
            transforms.RandomVerticalFlip(p=self.vertical),
            transforms.RandomAutocontrast(p=self.autocontrast),
            transforms.RandomEqualize(p=self.equalize),
            transforms.RandomRotation(degrees=(-1 * self.rotation, self.rotation)),
            transforms.ToTensor(),
            utils.normalize_transform(),
        ])

    def run(self):
        self.thread_active = True
        if self.thread_active:
            self.enabled_learning_process.emit(True)
            train_dataset = Dataset(f"{self.dataset_name}/train/",
                                    transform=self.get_augment())
            test_dataset = Dataset(f"{self.dataset_name}/valid/")
            loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            # model = Model(self.annotation)

            self.learn.emit("\n")
            self.learn.emit(f"Device: {self.device}")
            self.learn.emit(f"Dataset name: {self.dataset_name}")
            self.learn.emit(f"Batch Size: {self.batch_size}")
            self.learn.emit(f"Annotation: {self.annotation}")
            self.learn.emit(f"Epochs: {self.epochs}")
            self.learn.emit(f"Lr Step Size: {self.lr_step_size}")
            self.learn.emit(f"Learning Rate: {self.learning_rate}")
            self.learn.emit(f"Model Name: {self.model_name}")

            # dashed = "-" * 100
            # equals = "=" * 50
            # self.learn.emit(f"{equals}")
            self.learn.emit("")

            self.learn.emit(f"Deep Learning in process")
            self.learn.emit("")

            def fit(dataset, val_dataset=None, epochs=10, learning_rate=0.005, momentum=0.9,
                    weight_decay=0.0005, gamma=0.1, lr_step_size=3, verbose=True):

                if verbose and self.device == torch.device('cpu'):
                    print('It looks like you\'re training your model on a CPU. '
                          'Consider switching to a GPU; otherwise, this method '
                          'can take hours upon hours or even days to finish. '
                          'For more information, see https://detecto.readthedocs.io/'
                          'en/latest/usage/quickstart.html#technical-requirements')

                # Convert dataset to data loader if not already
                if not isinstance(dataset, DataLoader):
                    dataset = DataLoader(dataset, shuffle=True)

                if val_dataset is not None and not isinstance(val_dataset, DataLoader):
                    val_dataset = DataLoader(val_dataset)

                data = []
                losses = []
                # Get parameters that have grad turned on (i.e. parameters that should be trained)
                parameters = [par for par in self._model.parameters() if par.requires_grad]
                # Create an optimizer that uses SGD (stochastic gradient descent) to train the parameters
                optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum,
                                            weight_decay=weight_decay)
                # Create a learning rate scheduler that decreases learning rate by gamma every lr_step_size epochs
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

                # Train on the entire dataset for the specified number of times (epochs)
                for epoch in range(epochs):
                    if self.thread_active is False:
                        break

                    # status emit deep learning process
                    self.status.emit(f"Deep Learning in process")

                    if verbose:
                        print('Epoch {} of {}'.format(epoch + 1, epochs))

                    # Training step
                    self._model.train()

                    if verbose:
                        print('Begin iterating over training dataset')

                    iterable = tqdm(
                        dataset,
                        position=0,
                        leave=True,
                        bar_format="{desc:<4}{bar:30}{percentage:3.0f}% {r_bar}") if verbose else dataset

                    for images, targets in iterable:
                        if self.thread_active is False:
                            break
                        self.convert_to_int_labels(targets)
                        images, targets = self.to_device(images, targets)

                        # Calculate the model's loss (i.e. how well it does on the current
                        # image and target, with a lower loss being better)
                        loss_dict = self._model(images, targets)
                        total_loss = sum(loss for loss in loss_dict.values())

                        # Zero any old/existing gradients on the model's parameters
                        optimizer.zero_grad()
                        # Compute gradients for each parameter based on the current loss calculation
                        total_loss.backward()
                        # Update model parameters from gradients: param -= learning_rate * param.grad
                        optimizer.step()
                        self.status.emit(f"Iterating over training set: {iterable}")

                    # Validation step
                    if val_dataset is not None:
                        avg_loss = 0
                        with torch.no_grad():
                            if verbose:
                                print('Begin iterating over validation dataset')

                            iterable = tqdm(
                                val_dataset,
                                position=0,
                                leave=True,
                                bar_format="{desc:<4}{bar:30}{percentage:3.0f}% {r_bar}") if verbose else val_dataset
                            for images, targets in iterable:
                                if self.thread_active is False:
                                    break
                                self.convert_to_int_labels(targets)
                                images, targets = self.to_device(images, targets)
                                loss_dict = self._model(images, targets)
                                total_loss = sum(loss for loss in loss_dict.values())
                                avg_loss += total_loss.item()
                                self.status.emit(f"Iterating over validation set: {iterable}")

                        if avg_loss != 0:
                            avg_loss /= len(val_dataset.dataset)
                        losses.append(avg_loss)

                        if verbose:
                            print('Loss: {}'.format(avg_loss))

                        if self.thread_active is True:
                            self.learn.emit(
                                f"Epoch: {epoch + 1:003d}/{epochs:003d}\tLoss: {avg_loss:0005.4f}")
                            # self.learn.emit(f"{dashed}")
                            data.append((epoch + 1, avg_loss))
                            self.learn_graph.emit(data)

                    # Update the learning rate every few epochs
                    lr_scheduler.step()

                if len(losses) > 0:
                    return losses

            # self.learn.emit(f"{dashed}")

            fit(
                dataset=loader,
                val_dataset=test_dataset,
                epochs=self.epochs,
                lr_step_size=self.lr_step_size,
                learning_rate=self.learning_rate,
                verbose=True
            )

            self.learn.emit("")
            # self.learn.emit(f"{equals}")
            self.learn.emit("")

            if self.thread_active is True:
                self.save(os.path.join(f"../Models/{self.model_name}.pth"))
                self.status.emit(f"The model {self.model_name}.pth has saved.")
                self.learn.emit(f"The model {self.model_name}.pth has saved")
                with open(os.path.join(f"../Models/{self.model_name}.pth.txt"), "w") as f:
                    [f.writelines(f"{line}\n") for line in self.annotation]
                f.close()
                self.status.emit(f"The annotation {self.model_name}.pth.txt has saved")
                self.learn.emit(f"The annotation {self.model_name}.pth.txt has saved")

                self.learn.emit("")
                # self.learn.emit(f"{equals}")
                self.learn.emit("")

                self.learn.emit(f"Deep Learning process has been finished")
                self.status.emit(f"Deep Learning process has been finished")
            else:
                self.status_interrupt()

            self.stop()

    def stop(self):
        self.enabled_learning_process.emit(False)
        self.thread_active = False
        self.quit()