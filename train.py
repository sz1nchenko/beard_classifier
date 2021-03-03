import os
import warnings
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything

from models import Resnet, EfficientNet
from data import BeardDataModule
from config import NUM_CLASSES, SEED, BATCH_SIZE, DATA_PATH

warnings.filterwarnings('ignore')


def get_model(model_name, num_classes):
    model = None
    if 'resnet' in model_name:
        model = Resnet(model_name, num_classes)
    elif 'efficientnet' in model_name:
        model = EfficientNet(model_name, num_classes)
    else:
        raise ValueError(f'Undefined model name: {model_name}')

    return model


if __name__ == '__main__':
    seed_everything(SEED)

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--checkpoints', type=str, required=True, help='Directory name where to save checkpoints')
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to data directory')
    parser.add_argument('--tune', type=bool, default=False, help='Tune model before training (find LR and batch size )')
    parser.add_argument('--log_dir', type=str, default='default', help='Log directory name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    model = get_model(args.model, NUM_CLASSES)

    dm = BeardDataModule(data_path=args.data_path, batch_size=args.batch_size)

    if args.tune:
        trainer = pl.Trainer(auto_scale_batch_size=True, auto_lr_find=True)
        trainer.tune(model=model, datamodule=dm)

    logger = TensorBoardLogger(save_dir=os.getcwd(), name=args.log_dir)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join(os.getcwd(), 'checkpoints', args.checkpoints),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min',
        verbose=True
    )
    early_stoping_callback = EarlyStopping('val_loss', patience=7)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = logger
    trainer.callbacks = [checkpoint_callback, early_stoping_callback]
    trainer.fit(model=model, datamodule=dm)
    print('Best model with loss {:.4f} located in {}'.format(
        checkpoint_callback.best_model_score, checkpoint_callback.best_model_path)
    )
    trainer.test()