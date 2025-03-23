#import open3d as o3d
from rinarak.benchmarks.vqa.shapes3 import Shapes3Dataset
from rinarak import stprint

train_dataset = Shapes3Dataset(dataset_size=1024)  # create a dataset with 1024 samples
test_dataset = Shapes3Dataset(dataset_size=128)  # create a dataset with 128 samples for testing

# Now let's visualize a few samples from the dataset:
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(train_dataset[i]['image'].permute(1, 2, 0).numpy()[..., ::-1] * 0.5 + 0.5)
    plt.title(train_dataset[i]['question'] + ': ' + str(train_dataset[i]['answer']))
    stprint(train_dataset[i])
plt.tight_layout()
plt.show()

