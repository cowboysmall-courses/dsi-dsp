
import matplotlib.pyplot as plt
import seaborn as sns



def sns_setup(sns_style = "darkgrid", sns_context = "paper"):
    sns.set_style(sns_style)
    sns.set_context(sns_context)



def bar_plot(x_vals, y_vals, x_label, y_label, description):
    sns.barplot(x = x_vals, y = y_vals)
    plt.title(f"Bar Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def box_plot(x_vals, y_vals, x_label, y_label, description):
    sns.boxplot(x = x_vals, y = y_vals)
    plt.title(f"Box Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def box_plot_values(values, x_label, y_label, description):
    sns.boxplot(data = values)
    plt.title(f"Box Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def histogram(values, x_label, y_label, description):
    sns.histplot(values, kde = True)
    plt.title(f"Histogram: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def line_plot(x_vals, y_vals, x_label, y_label, description):
    sns.lineplot(x = x_vals, y = y_vals)
    plt.title(f"Line Plot: {description}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()



def correlation_matrix(data, description):
    ax = sns.heatmap(data, annot = True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 30,
        horizontalalignment = "right"
    )
    plt.title(f"Heat Map: {description}")
    plt.show()



def heat_map(data, description):
    sns.heatmap(data)
    plt.title(f"Heat Map: {description}")
    plt.show()
