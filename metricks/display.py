
def display_metrics_from_generator( model, generator ):
    """ Prints neatly the metrics of a model evaluated from a data generator. """
    metrics = model.evaluate_generator( generator )
    max_len = 0
    for name in model.metrics_names:
        if len(name) > max_len:
            max_len = len(name)
    for name, value in zip(model.metrics_names, metrics):
        print(name, " "*(3+max_len-len(name)), value)

def history_plot( history, metric, width=12, height=12 ):
    """ Plots the training and validation values of a metrics from a keras training log. """
    import pandas as pd
    import matplotlib.pyplot as plt

    history = pd.DataFrame(history.history)

    plt.style.use("ggplot")

    plt.figure(figsize=(width,height));
    plt.plot(history[metric], label="training "+metric);
    plt.plot(history["val_"+metric], label="validation "+metric);
    plt.title(metric);
    plt.legend(loc="best")
    plt.show();