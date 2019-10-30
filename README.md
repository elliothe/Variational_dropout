# Variational Dropout
Implementation of "Variational Dropout and the Local Reparameterization Trick" paper with Pytorch

This repository is modified from https://github.com/kefirski/variational_dropout , where some pytorch version bug are addressed.

```bash
$ python train.py --mode [simple/dropout/vardropout] --data_path [your-mnist-path]
```

## Experiment

The validation loss with 60 epochs are listed as follows, after the model trained with the provided `train.py`:

| Scheme              | val loss | notes                                         |
| ------------------- | -------- | --------------------------------------------- |
| simple              | 0.0814   | slight overfitting after training is finished |
| dropout             | 0.0714   | best result, reach 0.061 after 100 epoch      |
| variational dropout | 0.07813  | train 100 epochs can reduce the loss 0.069   |