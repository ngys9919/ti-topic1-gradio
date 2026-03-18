# Run MNIST CNN Trainer

This plan outlines the steps to launch the Gradio application for the MNIST CNN Trainer.

## Proposed Changes

### Execution
- Launch the application using `.venv\Scripts\python.exe main.py`.
- The application will start a local Gradio server.

## Verification Plan

### Automated Tests
- Check the terminal output for the local URL (usually `http://127.0.0.1:7860`).
- Use the browser tool to navigate to the URL and verify the Gradio interface loads correctly.

### Manual Verification
- The user can interact with the Gradio interface to train the model.
