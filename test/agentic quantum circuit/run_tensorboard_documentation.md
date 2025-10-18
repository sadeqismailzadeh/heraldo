# Documentation for `run_tensorboard.py`

## 1. Overview

This is a simple utility script whose sole purpose is to launch and manage **TensorBoard**, a powerful web-based visualization toolkit that is part of the TensorFlow ecosystem.

`stable-baselines3` integrates with TensorBoard to log a wide variety of training metrics. This script provides a convenient way to view these metrics in real-time.

## 2. Purpose and Functionality

-   **What does it do?**: It starts a local web server that serves the TensorBoard user interface and points it to the directory where the training logs are stored.
-   **Why is it useful?**: During a long training run (as performed by `train_quantum_circuit.py`), it is crucial to monitor the agent's progress. TensorBoard provides live-updating plots of key metrics, allowing you to answer questions like:
    -   Is the agent learning? (Is the `rollout/ep_rew_mean`, the mean reward, increasing?)
    -   How well is it performing on the main objective? (Is the `custom/final_fidelity` increasing?)
    -   Is the learning process stable? (Are the `train/policy_loss` and `train/value_loss` decreasing smoothly?)
    -   How much is the agent exploring? (What is the `train/entropy_loss` doing?)

## 3. How It Works

1.  **Import Libraries**: It imports `os` for path manipulation, `webbrowser` to automatically open a browser tab, `time` for the main loop, and `program` from the `tensorboard` library.
2.  **Set Log Directory**: It defines the `log_dir` variable, pointing it to the `Train` folder. This **must** be the same directory that is passed to the `tensorboard_log` argument of the PPO model in `train_quantum_circuit.py`.
3.  **Configure and Launch TensorBoard**:
    -   `tb = program.TensorBoard()`: Creates an instance of the TensorBoard program.
    -   `tb.configure(argv=[None, '--logdir', log_dir])`: Configures it, programmatically passing the command-line argument `--logdir` to tell it where to find the log files.
    -   `url = tb.launch()`: Starts the TensorBoard server. The `launch()` method returns the URL of the local server (e.g., `http://localhost:6006/`).
4.  **Open in Browser**: `webbrowser.open(url)` automatically opens this URL in the user's default web browser, making it easy to access the interface.
5.  **Keep Script Alive**: The `while True: time.sleep(1)` loop keeps the Python script running. If the script were to exit, the TensorBoard server it launched would also be terminated. The script can be stopped by pressing `Ctrl+C` in the terminal.

## 4. How to Use

1.  Start the training process by running `python train_quantum_circuit.py` in one terminal.
2.  Once training has started and the `Train` directory has been created, open a **second terminal**.
3.  In the second terminal, run this script: `python run_tensorboard.py`.
4.  A new tab should open in your web browser displaying the TensorBoard interface with live plots of your training run.

## 5. Flowchart of Script Execution

```mermaid
graph TD
    A[Start Script] --> B[Set log_dir = './Train'];
    B --> C[Instantiate TensorBoard Program];
    C --> D[Configure TensorBoard to use log_dir];
    D --> E[Launch TensorBoard Server];
    E --> F[Get Server URL (e.g., http://localhost:6006)];
    F --> G[Open URL in Web Browser];
    G --> H[Enter Infinite Loop to Keep Server Alive];
    H -- Ctrl+C --> I[Stop Script and Server];
```
