
import matplotlib.pyplot as plt



def plot_setup(style = "ggplot", figsize = None, tight_layout = False):
    if figsize:
        plt.figure(figsize = figsize)
    if tight_layout:
        plt.tight_layout()
    plt.style.use(style)



def scatter_plot(x_vals, y_vals, x_label, y_label, description):
    plt.scatter(x_vals, y_vals)
    plt.title(f"Scatter Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def roc_curve(fpr, tpr, description):
    plt.plot(fpr, tpr, label = 'ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f"ROC Curve: {description}")
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc = 'lower right')
    plt.show()



def horizontal_bar_plot(x_vals, y_vals, x_label, y_label, description):
    plt.barh(x_vals, y_vals, align = 'center')
    plt.title(f"Horizontal Bar Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def image_plot(image):
    plt.imshow(image, interpolation = "bilinear")
    plt.axis("off")
    plt.show()
