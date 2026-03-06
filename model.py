from sklearn.metrics import roc_auc_score
from torchmetrics import Accuracy, Recall
import pytorch_lightning as pl
import pandas as pd
import timm  
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from dataloader import *
from utils_sampling import *
from data_split import *
import logging
import argparse
import os
import time
import subprocess

logging.basicConfig(filename='training.log',filemode='w',level=logging.INFO, force=True)

class ImageClassifier(pl.LightningModule):
    def __init__(self, lmd=0):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)
        self.accuracy = Accuracy(task='binary', threshold=0.5)
        self.recall = Recall(task='binary', threshold=0.5)  
        self.validation_outputs = []
        self.lmd = lmd
        self.last_batch_end_time = None
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        if self.last_batch_end_time is not None:
            data_time = now - self.last_batch_end_time
            logging.info(f"Data time before batch {batch_idx}: {data_time:.3f}s")
        self.step_start_time = now
    
    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        now = time.perf_counter()
        step_time = now - self.step_start_time
        self.last_batch_end_time = now

        gpu_util, mem_util = "NA", "NA"
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu, utilization.memory",
                    "--format=csv,noheader,nounits,"
                ]
            capture_output = True,
            text = True,
            check = True
            )
            gpu_util, mem_util = result.stdout.srip().split(", ")
        except Exception as e:
            logging.warning(f"Failed to query nvidia-smi: {e}")

        logging.info(
            f"Compute time for batch {batch_idx} took {step_time:.3f}s,"
            f"GPU util {gpu_util}% mem util {mem_util}%"
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()
        
        print(f"Shape of outputs (training): {outputs.shape}")
        print(f"Shape of labels (training): {labels.shape}")
        
        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        logging.info(f"Training Step - ERM loss: {loss.item()}")
        loss += self.lmd * (outputs ** 2).mean() # SD loss penalty
        logging.info(f"Training Step - SD loss: {loss.item()}")
        return loss

    def validation_step(self, batch):
        images, labels, _ = batch
        outputs = self.forward(images).squeeze()

        if outputs.shape == torch.Size([]):
            return
        
        print(f"Shape of outputs (validation): {outputs.shape}")
        print(f"Shape of labels (validation): {labels.shape}")

        loss = F.binary_cross_entropy_with_logits(outputs, labels.float())
        preds = torch.sigmoid(outputs)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.accuracy(preds, labels.int()), prog_bar=True, sync_dist=True)
        self.log('val_recall', self.recall(preds, labels.int()), prog_bar=True, sync_dist=True)
        output = {"val_loss": loss, "preds": preds, "labels": labels}
        self.validation_outputs.append(output)
        logging.info(f"Validation Step - Batch loss: {loss.item()}")
        return output
    
    def predict_step(self, batch):
        images, label, domain = batch
        outputs = self.forward(images).squeeze()
        preds = torch.sigmoid(outputs)
        return preds, label, domain

    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            logging.warning("No outputs in validation step to process")
            return
        preds = torch.cat([x['preds'] for x in self.validation_outputs])
        labels = torch.cat([x['labels'] for x in self.validation_outputs])
        if labels.unique().size(0) == 1:
            logging.warning("Only one class in validation step")
            return
        auc_score = roc_auc_score(labels.cpu(), preds.cpu())
        self.log('val_auc', auc_score, prog_bar=True, sync_dist=True)
        logging.info(f"Validation Epoch End - AUC score: {auc_score}")
        self.validation_outputs = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        return optimizer


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./model_checkpoints/',
    filename='image-classifier-{step}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    every_n_train_steps=1001,
    enable_version_counter=True
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=4,
    mode="min",
)

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", help="checkpoint to continue from", required=False)
parser.add_argument("--predict", help="predict on test set", action="store_true")
parser.add_argument("--reset", help="reset training", action="store_true")
args = parser.parse_args()

train_domains = [0, 1, 4]
val_domains = [0, 1, 4]
lmd_value = 0

if args.predict:
    test_dl = load_dataloader([0, 1, 2, 3, 4], "test", batch_size=32, num_workers=8)
    model = ImageClassifier.load_from_checkpoint(args.ckpt_path)
    trainer = pl.Trainer()
    predictions = trainer.predict(model, dataloaders=test_dl)
    preds, labels, domains = zip(*predictions)
    preds = torch.cat(preds).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    domains = torch.cat(domains).cpu().numpy()
    print(preds.shape, labels.shape, domains.shape)
    df = pd.DataFrame({"preds": preds, "labels": labels, "domains": domains})
    filename = "preds-" + args.ckpt_path.split("/")[-1]
    df.to_csv(f"outputs/{filename}.csv", index=False)
else:
    train_dl = load_dataloader(train_domains, "train", batch_size=32, num_workers=8)
    logging.info("Training dataloader loaded")
    val_dl = load_dataloader(val_domains, "val", batch_size=32, num_workers=8)
    logging.info("Validation dataloader loaded")

    if args.reset:
        model = ImageClassifier.load_from_checkpoint(args.ckpt_path)
    else:
        model = ImageClassifier(lmd=lmd_value)
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, early_stop_callback],
        max_steps=20000,
        val_check_interval=1000,
        check_val_every_n_epoch=None
    )
    trainer.fit(
        model=model, 
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=args.ckpt_path if not args.reset else None
    )
