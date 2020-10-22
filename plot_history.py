import numpy as np
import matplotlib.pyplot as plt

def plot_model(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # accuracy history 요약
    axs[0].plot(range(1, len(model_history.history["accuracy"])+1, model_history.history["accuracy"]))
    axs[0].plot(range(1, len(model_history.history["test_accuracy"])+1, model_history.history["val_accuracy"]))

    axs[0].set_title("Model Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epoch")

    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)

    axs[0].legend(["train", "test"], loc="best")

    # loss history 요약
    axs[1].plot(range(1, len(model_history.history["loss"]+1), model_history.history["loss"]))
    axs[1].plot(range(1, len(model_history.history["test_loss"]+1), model_history.history["test_loss"]))

    axs[1].set_title("Model Loss")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(["train", "test"], loc="best")

    fig.savefig("plot.png")
    plt.show()