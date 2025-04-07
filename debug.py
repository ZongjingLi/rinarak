from tqdm.gui import tqdm
import time

# Basic usage with an iterable
for i in tqdm(range(100)):
    time.sleep(0.05)  # Some time-consuming operation

# With a manually managed progress bar
with tqdm(total=100) as pbar:
    for i in range(100):
        time.sleep(0.05)
        pbar.update(1)

# Specifying additional parameters
for i in tqdm(range(100), desc="Processing", unit="items", colour="green"):
    time.sleep(0.05)