import os
import torchvision
import torch
import logging
from src.external import qtc
from detecto import utils
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from .rcnn_model import Dataset, DataLoader, extract_labels_from_xml_folder
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WorkerRCNN(qtc.QThread):
    """
    deep learning process
    """
    learn = qtc.pyqtSignal(str)
    learn_graph = qtc.pyqtSignal(list)
    status = qtc.pyqtSignal(str)
    exception = qtc.pyqtSignal(str)
    enabled_learning_process = qtc.pyqtSignal(bool)

    def __init__(self, dataset_name: str, batch_size: int, annotation: list, epochs: int,
                 lr_step_size: int, learning_rate: float, model_name: str):
        super(WorkerRCNN, self).__init__()
        self.thread_active = False
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        # Normalize annotation: accept either a list (['A','B']) or a comma-separated
        # string like "A, B" or 'A, B' and produce a list of label strings.
        if isinstance(annotation, str):
            ann_list = [s.strip().strip('"').strip("'") for s in annotation.split(',') if s.strip()]
            self.annotation = ann_list
        elif annotation is None:
            self.annotation = []
        else:
            # assume iterable of labels
            self.annotation = list(annotation)
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
        # use torch.device for consistency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # Use new weights API when available to avoid deprecation warnings
        try:
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        except Exception:
            # fallback to older API
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.eval = self._model.eval()
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

        # Optional post-training exports (kept off by default to avoid changing GUI behavior)
        self.export_onnx: bool = False
        self.quantize_onnx_int8: bool = False

    def status_interrupt(self):
        self.learn.emit("\tDeep Learning has been interrupted!")
        self.exception.emit("Deep Learning has been interrupted!")

    # Converts all string labels in a list of target dicts to
    # their corresponding int mappings
    def convert_to_int_labels(self, targets):
        for idx, target in enumerate(targets):
            # get all string labels for objects in a single image
            labels_array = target['labels']
            # detect labels missing from the mapping
            missing = sorted({c for c in labels_array if c not in self._int_mapping})
            if missing:
                # fail early: dataset contains labels not declared in annotation
                raise ValueError(f"Dataset contains label(s) not present in annotation: {missing}. Update annotation or dataset.")
            # convert string labels into integer labels
            labels_int_array = [self._int_mapping[class_name] for class_name in labels_array]
            target['labels'] = torch.tensor(labels_int_array, dtype=torch.int64)

    def validate_dataset_labels(self, dataloader):
        """Validate that all labels present in dataloader are listed in self.annotation.
        Raises ValueError when unknown labels are found.
        """
        dataset_labels = set()
        # dataloader can be an iterable over (images, targets)
        for _, targets in dataloader:
            for t in targets:
                # t['labels'] may be a list of strings (before convert_to_int_labels)
                labels = t.get('labels', [])
                dataset_labels.update(labels)
        missing = sorted(dataset_labels - set(self._classes))
        if missing:
            raise ValueError(f"Dataset contains label(s) not present in annotation: {missing}")

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

    def set_export_onnx(self, enabled: bool):
        self.export_onnx = bool(enabled)

    def set_quantize_onnx_int8(self, enabled: bool):
        self.quantize_onnx_int8 = bool(enabled)

    @staticmethod
    def _add_prob_transform(transforms_list, p, factory):
        """Append a torchvision transform created by factory(p) when p is a positive number."""
        if isinstance(p, (int, float)) and p > 0:
            transforms_list.append(factory(p))

    def _build_transforms_list(self):
        if self.resize is None:
            raise ValueError("Resize not set. Call set_resize(resize) before get_augment().")

        t = [transforms.ToPILImage(), transforms.Resize((self.resize, self.resize))]
        self._add_prob_transform(t, self.horizontal, lambda p: transforms.RandomHorizontalFlip(p=p))
        self._add_prob_transform(t, self.vertical, lambda p: transforms.RandomVerticalFlip(p=p))
        self._add_prob_transform(t, self.autocontrast, lambda p: transforms.RandomAutocontrast(p=p))
        self._add_prob_transform(t, self.equalize, lambda p: transforms.RandomEqualize(p=p))
        if isinstance(self.rotation, (int, float)) and self.rotation > 0:
            t.append(transforms.RandomRotation(degrees=(-1 * self.rotation, self.rotation)))
        t.append(transforms.ToTensor())
        t.append(utils.normalize_transform())
        return t

    def get_augment(self):
        """Build and return the augmentation transforms based on the configured
        augmentation parameters. Only active transforms (where params are not None)
        are included; this avoids passing None into torchvision transforms.
        """
        transforms_list = self._build_transforms_list()
        self.custom_transforms = transforms.Compose(transforms_list)
        return self.custom_transforms

    def _ensure_dataloaders(self, dataset, val_dataset):
        if not isinstance(dataset, DataLoader):
            dataset = DataLoader(dataset, shuffle=True)
        if val_dataset is not None and not isinstance(val_dataset, DataLoader):
            val_dataset = DataLoader(val_dataset)
        return dataset, val_dataset

    def _make_optimizer(self, learning_rate, momentum, weight_decay, lr_step_size, gamma):
        params = [par for par in self._model.parameters() if par.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)
        return optimizer, scheduler

    def _train_one_epoch(self, dataset, optimizer, verbose):
        self._model.train()
        iterable = tqdm(dataset, position=0, leave=True,
                        bar_format="{desc:<4}{bar:30}{percentage:3.0f}% {r_bar}") if verbose else dataset
        for images, targets in iterable:
            if self.thread_active is False:
                break
            self.convert_to_int_labels(targets)
            images, targets = self.to_device(images, targets)
            loss_dict = self._model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            self.status.emit(f"Iterating over training set: {iterable}")

    def _validate_one_epoch(self, val_dataset, verbose):
        if val_dataset is None:
            return None
        avg_loss = 0.0
        with torch.no_grad():
            iterable = tqdm(val_dataset, position=0, leave=True,
                            bar_format="{desc:<4}{bar:30}{percentage:3.0f}% {r_bar}") if verbose else val_dataset
            for images, targets in iterable:
                if self.thread_active is False:
                    break
                self.convert_to_int_labels(targets)
                images, targets = self.to_device(images, targets)
                loss_dict = self._model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                avg_loss += float(total_loss.item())
                self.status.emit(f"Iterating over validation set: {iterable}")

        if len(val_dataset.dataset) == 0:
            return 0.0
        return avg_loss / len(val_dataset.dataset)

    def fit(self, dataset, val_dataset=None, epochs=10, learning_rate=0.005, momentum=0.9,
            weight_decay=0.0005, gamma=0.1, lr_step_size=3, verbose=False):

        if verbose and self.device == torch.device('cpu'):
            self.learn.emit('It looks like you\'re training your model on a CPU. '
                  'Consider switching to a GPU; otherwise, this method '
                  'can take hours upon hours or even days to finish. '
                  'For more information, see https://detecto.readthedocs.io/'
                  'en/latest/usage/quickstart.html#technical-requirements')

        dataset, val_dataset = self._ensure_dataloaders(dataset, val_dataset)
        data = []
        losses = []
        optimizer, lr_scheduler = self._make_optimizer(learning_rate, momentum, weight_decay, lr_step_size, gamma)

        # Train on the entire dataset for the specified number of times (epochs)
        for epoch in range(epochs):
            if self.thread_active is False:
                break

            # status emit deep learning process
            self.status.emit(f"Deep Learning in process")

            self._train_one_epoch(dataset, optimizer, verbose)

            avg_loss = self._validate_one_epoch(val_dataset, verbose)
            if avg_loss is not None and avg_loss != 0:
                avg_loss /= len(val_dataset.dataset)
            losses.append(avg_loss)

            # if verbose:
            # print('Loss: {}'.format(avg_loss))

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

    def run(self):
        self.thread_active = True
        if self.thread_active:
            self.enabled_learning_process.emit(True)
            train_dataset = Dataset(f"{self.dataset_name}/train/",
                                    transform=self.get_augment())
            test_dataset = Dataset(f"{self.dataset_name}/valid/")
            loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            # Validate XML annotations (keep XML as single source of truth)
            train_xml_folder = os.path.join(self.dataset_name, 'train')
            try:
                xml_labels = extract_labels_from_xml_folder(train_xml_folder)
                unknown_xml_labels = sorted(xml_labels - set(self._classes))
                if unknown_xml_labels:
                    msg = f"Found labels in XML not in annotation: {unknown_xml_labels}. Update annotation or edit XMLs."
                    self.learn.emit(msg)
                    self.exception.emit(msg)
                    raise ValueError(msg)
            except Exception as e:
                # emit and re-raise to stop training
                logger.exception("Error while validating XML labels: %s", e)
                raise

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
            self.learn.emit("\n")
            self.learn.emit("=" * 30)
            self.learn.emit("\n")
            self.learn.emit(str(self.eval))
            self.learn.emit("=" * 30)
            self.learn.emit("\n")
            self.learn.emit("Deep Learning in process")
            self.learn.emit("=" * 30)
            # self.learn.emit(f"{dashed}")

            self.fit(
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
                # Save annotations as a single comma-separated line (no surrounding brackets/quotes)
                ann_file = os.path.join(f"../Models/{self.model_name}.pth.txt")
                os.makedirs(os.path.dirname(ann_file), exist_ok=True)
                with open(ann_file, "w", encoding="utf-8") as f:
                    f.write(", ".join(self.annotation))

                self.status.emit(f"The annotation {self.model_name}.pth.txt has saved")
                self.learn.emit(f"The annotation {self.model_name}.pth.txt has saved")

                # Optional ONNX export + INT8 quantization
                onnx_path, onnx_int8_path = self._onnx_paths()
                if self.export_onnx:
                    self.learn.emit(f"Exporting ONNX model to {onnx_path}...")
                    self._export_to_onnx(onnx_path)
                    if self.quantize_onnx_int8:
                        self.learn.emit(f"Exporting INT8 quantized ONNX model to {onnx_int8_path}...")
                        self._quantize_onnx_dynamic_int8(onnx_path, onnx_int8_path)

                self.learn.emit("")
                # self.learn.emit(f"{equals}")
                self.learn.emit("")

                self.learn.emit("Deep Learning process has been finished")
                self.status.emit("Deep Learning process has been finished")
            else:
                self.status_interrupt()

            self.stop()

    def stop(self):
        self.enabled_learning_process.emit(False)
        self.thread_active = False
        self.quit()

    def _onnx_paths(self) -> tuple[str, str]:
        """Return (onnx_fp32_path, onnx_int8_path) under Models/."""
        onnx_path = os.path.join(f"../Models/{self.model_name}.onnx")
        onnx_int8_path = os.path.join(f"../Models/{self.model_name}.int8.onnx")
        return onnx_path, onnx_int8_path

    def _export_to_onnx(self, out_path: str) -> None:
        """Export the trained torch model to ONNX.

        Notes:
        - This exports a torchvision FasterRCNN model.
        - ONNX output schema can vary; ONNX inference decoding is handled elsewhere.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        model = self._model
        model.eval()

        # Export is generally more stable on CPU (especially for tracing)
        export_device = torch.device('cpu')
        model_cpu = model.to(export_device)

        dummy = torch.zeros((1, 3, 600, 800), dtype=torch.float32, device=export_device)

        with torch.no_grad():
            torch.onnx.export(
                model_cpu,
                dummy,
                out_path,
                opset_version=11,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['outputs'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                },
            )

        # Move back to training/inference device for any subsequent operations
        model_cpu.to(self.device)

    def _emit_warning_messages(self, warnings_list):
        for ww in warnings_list:
            try:
                self.learn.emit(str(ww.message))
            except Exception:
                pass

    def _emit_info(self, msg: str):
        try:
            self.learn.emit(msg)
        except Exception:
            pass

    def _try_quantize_dynamic(self, quantize_dynamic, QuantType, in_path: str, out_path: str):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            quantize_dynamic(
                model_input=in_path,
                model_output=out_path,
                weight_type=QuantType.QInt8,
            )
        self._emit_warning_messages(w)

    def _quantize_onnx_dynamic_int8(self, in_path: str, out_path: str) -> None:
        """Create an INT8 dynamically-quantized ONNX model (CPU oriented).

        FasterRCNN graphs often contain ops that ONNXRuntime can't fully infer for
        quantization (e.g., NonZero). We treat quantization as best-effort.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as e:
            raise RuntimeError(f"onnxruntime quantization is not available: {e}")

        try:
            self._try_quantize_dynamic(quantize_dynamic, QuantType, in_path, out_path)
        except Exception as e:
            self._emit_info(f"ONNX INT8 quantization failed: {e}")
