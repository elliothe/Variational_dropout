# Variational Dropout
Implementation of "Variational Dropout and the Local Reparameterization Trick" paper with Pytorch

This repository is modified from https://github.com/kefirski/variational_dropout , where some pytorch version bug are addressed.

## Experiment


Those are the validation loss, after the model trained with the provided `train.py`.

| Scheme              | val loss | notes                                                             |
| ------------------- | -------- | ----------------------------------------------------------------- |
| simple              | 0.0814   | slight overfitting after training is finished                     |
| dropout             | 0.0714   | best result                                                       |
| variational dropout | 0.07813  | seems loss is still degrading, could train for more epochs to see |