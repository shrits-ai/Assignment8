# Custom CNN for CIFAR-10 Classification

This project implements a custom convolutional neural network for CIFAR-10 classification using PyTorch.

## Requirements

- Python 3.x
- Install dependencies using:
  ```bash
  pip install -r requirements.txt

Criteria:
- network has the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead
- total RF must be more than 44
- one of the layers must use Depthwise Separable Convolution
- one of the layers must use Dilated Convolution
- use GAP (compulsory):- add FC after GAP to target #of classes (optional)
- use augmentation library and apply:
horizontal flip
shiftScaleRotate
coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
- achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.


Result
- Achieved 85% accuracy after 35th epoch
- Total params: 199,442
  Epoch 1/20

Training: 100%|██████████████████████████████████████████████████████████████████| 782/782 [13:58<00:00,  1.07s/it, loss=0.58]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:39<00:00,  1.57it/s, acc=81.9, loss=0.545]

Train Loss: 0.5796, Train Acc : 80.8920, Val Loss: 0.5451, Val Acc: 81.88%

Checkpoint saved at epoch 1

Epoch 2/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:12<00:00,  1.01s/it, loss=0.572]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.64it/s, acc=83.2, loss=0.501]

Train Loss: 0.5716, Train Acc : 81.2400, Val Loss: 0.5008, Val Acc: 83.22%

Checkpoint saved at epoch 2

Epoch 3/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [12:57<00:00,  1.01it/s, loss=0.571]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.64it/s, acc=82.4, loss=0.534]

Train Loss: 0.5707, Train Acc : 81.1040, Val Loss: 0.5341, Val Acc: 82.43%

Checkpoint saved at epoch 3

Epoch 4/20

Training: 100%|██████████████████████████████████████████████████████████████████| 782/782 [13:04<00:00,  1.00s/it, loss=0.57]

Validation: 100%|███████████████████████████████████████████████████████| 157/157 [01:34<00:00,  1.66it/s, acc=82, loss=0.541]

Train Loss: 0.5702, Train Acc : 81.3100, Val Loss: 0.5413, Val Acc: 82.01%

Checkpoint saved at epoch 4

Epoch 5/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:01<00:00,  1.00it/s, loss=0.559]

Validation: 100%|████████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.65it/s, acc=84, loss=0.49]

Train Loss: 0.5593, Train Acc : 81.6640, Val Loss: 0.4896, Val Acc: 84.02%

Checkpoint saved at epoch 5

Epoch 6/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:16<00:00,  1.02s/it, loss=0.563]

Validation: 100%|██████████████████████████████████████████████████████| 157/157 [01:38<00:00,  1.59it/s, acc=83.3, loss=0.52]

Train Loss: 0.5635, Train Acc : 81.4660, Val Loss: 0.5200, Val Acc: 83.30%

Epoch 7/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:52<00:00,  1.07s/it, loss=0.563]

Validation: 100%|██████████████████████████████████████████████████████| 157/157 [01:36<00:00,  1.63it/s, acc=82.2, loss=0.52]

Train Loss: 0.5626, Train Acc : 81.3520, Val Loss: 0.5199, Val Acc: 82.20%

Epoch 8/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [12:58<00:00,  1.00it/s, loss=0.563]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.65it/s, acc=83.7, loss=0.487]

Train Loss: 0.5632, Train Acc : 81.6060, Val Loss: 0.4874, Val Acc: 83.65%

Epoch 9/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [12:59<00:00,  1.00it/s, loss=0.558]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.65it/s, acc=83.6, loss=0.492]

Train Loss: 0.5581, Train Acc : 81.7340, Val Loss: 0.4924, Val Acc: 83.63%

Checkpoint saved at epoch 9

Epoch 10/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:51<00:00,  1.06s/it, loss=0.555]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:35<00:00,  1.65it/s, acc=83.6, loss=0.503]

Train Loss: 0.5549, Train Acc : 81.6600, Val Loss: 0.5033, Val Acc: 83.59%

Checkpoint saved at epoch 10

Epoch 11/20

Training: 100%|██████████████████████████████████████████████████████████████████| 782/782 [13:00<00:00,  1.00it/s, loss=0.56]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:38<00:00,  1.60it/s, acc=83.6, loss=0.503]

Train Loss: 0.5604, Train Acc : 81.7560, Val Loss: 0.5033, Val Acc: 83.63%

Epoch 12/20

Training: 100%|██████████████████████████████████████████████████████████████████| 782/782 [13:02<00:00,  1.00s/it, loss=0.56]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:34<00:00,  1.66it/s, acc=82.8, loss=0.509]

Train Loss: 0.5601, Train Acc : 81.6480, Val Loss: 0.5088, Val Acc: 82.79%

Epoch 13/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:04<00:00,  1.00s/it, loss=0.552]

Validation: 100%|██████████████████████████████████████████████████████| 157/157 [01:34<00:00,  1.66it/s, acc=83.7, loss=0.49]

Train Loss: 0.5518, Train Acc : 82.0060, Val Loss: 0.4902, Val Acc: 83.69%

Checkpoint saved at epoch 13

Epoch 14/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:08<00:00,  1.01s/it, loss=0.557]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:41<00:00,  1.55it/s, acc=83.8, loss=0.496]

Train Loss: 0.5574, Train Acc : 81.6320, Val Loss: 0.4956, Val Acc: 83.79%

Epoch 15/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:54<00:00,  1.07s/it, loss=0.553]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:38<00:00,  1.59it/s, acc=83.5, loss=0.503]

Train Loss: 0.5532, Train Acc : 81.9120, Val Loss: 0.5034, Val Acc: 83.52%

Epoch 16/20

Training: 100%|██████████████████████████████████████████████████████████████████| 782/782 [14:03<00:00,  1.08s/it, loss=0.55]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:39<00:00,  1.58it/s, acc=85.2, loss=0.463]

Train Loss: 0.5496, Train Acc : 82.0520, Val Loss: 0.4631, Val Acc: 85.17%

Checkpoint saved at epoch 16

Epoch 17/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [14:03<00:00,  1.08s/it, loss=0.544]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:38<00:00,  1.60it/s, acc=85.3, loss=0.461]

Train Loss: 0.5442, Train Acc : 82.2760, Val Loss: 0.4613, Val Acc: 85.35%

Checkpoint saved at epoch 17

Epoch 18/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:59<00:00,  1.07s/it, loss=0.549]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:39<00:00,  1.58it/s, acc=85.56, loss=0.46]

Train Loss: 0.5488, Train Acc : 82.0640, Val Loss: 0.4602, Val Acc: 85.57%

Epoch 19/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:59<00:00,  1.07s/it, loss=0.536]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:38<00:00,  1.59it/s, acc=86.2, loss=0.456]

Train Loss: 0.5362, Train Acc : 82.1840, Val Loss: 0.4565, Val Acc: 86.15%

Epoch 20/20

Training: 100%|█████████████████████████████████████████████████████████████████| 782/782 [13:53<00:00,  1.07s/it, loss=0.545]

Validation: 100%|█████████████████████████████████████████████████████| 157/157 [01:39<00:00,  1.58it/s, acc=85.9, loss=0.453]

Train Loss: 0.5445, Train Acc : 82.0060, Val Loss: 0.4535, Val Acc: 85.86%
