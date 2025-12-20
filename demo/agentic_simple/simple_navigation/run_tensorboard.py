import os
from tensorboard import program
import webbrowser
import time

# Set the log directory where your training data is saved
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_navigation_tensorboard")

# Start TensorBoard
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', log_dir])
url = tb.launch()
print(f"TensorBoard started at {url}")

# Open the TensorBoard interface in your default web browser
webbrowser.open(url)

try:
    print("Press Ctrl+C to stop TensorBoard")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping TensorBoard...")