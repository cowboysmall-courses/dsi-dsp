
import matplotlib.pyplot as plt



def plot_setup(figsize = (12, 9), style = "ggplot"):
    plt.figure(figsize = figsize)
    plt.tight_layout()
    plt.style.use(style)



def scatter_plot(data, column, column_name, index_name, sub_path):
    plt.clf()
    plt.scatter(data.index, data[column].values)
    plt.title(f"Scatter Plot: {index_name}")
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.savefig(f"./images/{sub_path}/scatter/{column}.png")
    plt.close()



def roc_curve(fpr, tpr, name, description, sub_path):
    plt.clf()
    plt.plot(fpr, tpr, label = 'ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"ROC Curve: {description}")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc = 'lower right')
    plt.savefig(f"./images/{sub_path}/roc/roc_curve_{name}.png")
    plt.close()
