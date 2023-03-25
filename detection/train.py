import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from transform import get_transform
from dataset import PennFudanDataset
from model import rcnn_detection_model
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset, convert_to_coco_api


class RCNNDetectionModule(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, momentum, weight_decay, step_size, gamma, val_dataset):
        super().__init__()
        self.model = rcnn_detection_model(num_classes)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.validation_step_outputs = []
        self.val_dataset = val_dataset

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        # log training losses
        self.log('train_loss', losses, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images, targets)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.validation_step_outputs.append({"predictions": res, "targets": targets})
        return {"predictions": res, "targets": targets}
    
    def on_validation_epoch_end(self):
        
        # Create the COCO format dataset
        coco = get_coco_api_from_dataset(self.val_dataset)
        # Combine all predictions and targets
        all_predictions = {}
        all_targets = []
        for output in self.validation_step_outputs:
            all_predictions.update(output["predictions"])
            all_targets.extend(output["targets"])
        
        # Create the CocoEvaluator
        coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

        # Evaluate the predictions
        coco_evaluator.update(all_predictions)
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        # Log IoU, mAP, and recall for the test set after each epoch
        iou = coco_evaluator.coco_eval["bbox"].stats[0]
        mAP = coco_evaluator.coco_eval["bbox"].stats[1]
        recall = coco_evaluator.coco_eval["bbox"].stats[8]

        self.log("val_iou", iou, prog_bar=True, logger=True)
        self.log("val_mAP", mAP, prog_bar=True, logger=True)
        self.log("val_recall", recall, prog_bar=True, logger=True)

        self.validation_step_outputs.clear()
        super().on_validation_epoch_end()

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        return [optimizer], [scheduler]

class PennFudanDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size, test_batch_size, num_workers, test_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.test_size = test_size

    def setup(self, stage=None):
        transform_train = get_transform(train=True)
        transform_test = get_transform(train=False)

        dataset = PennFudanDataset(self.data_path, transform_train)

        n_samples = len(dataset)
        train_size = int(0.9 * n_samples)
        test_size = n_samples - train_size
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, test_size])
        self.val_dataset = torch.utils.data.Subset(self.val_dataset, range(self.test_size))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=utils.collate_fn)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    NUM_CLASSES = 2
    DATASET_PATH = '/Users/shubhamrandive/Documents/datasets/PennFudanPed'
    BATCH_SIZE = 2
    TEST_BATCH_SIZE = 2
    NUM_WORKERS = 4
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 3
    GAMMA = 0.1
    NUM_EPOCHS = 1
    TEST_SIZE = 50
    PRINT_FREQ = 10

    

    data_module = PennFudanDataModule(DATASET_PATH, BATCH_SIZE, TEST_BATCH_SIZE, NUM_WORKERS, TEST_SIZE)

    #use limited data and limited epochs for testing 
    data_module.setup()

    model = RCNNDetectionModule(NUM_CLASSES, LEARNING_RATE, MOMENTUM, WEIGHT_DECAY, STEP_SIZE, GAMMA,data_module.val_dataset)
    model.to(device)

    # define trainer with fast_dev_run for testing
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, fast_dev_run=1)
    trainer.fit(model, data_module)

    print("That's it!")

if __name__ == '__main__':
    main()

