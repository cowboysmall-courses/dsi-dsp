
import matplotlib.pyplot as plt
import seaborn as sns



def sns_setup(sns_style = "darkgrid", sns_context = "paper"):
    sns.set_style(sns_style)
    sns.set_context(sns_context)



def bar_plot(x_vals, y_vals, column, column_name, interval, interval_name, index_name, sub_path):
    plt.clf()
    sns.barplot(x = x_vals, y = y_vals)
    plt.title(f"Bar Plot: {index_name}")
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig(f"./images/{sub_path}/barplot/{column}_{interval}.png")
    plt.close()



def box_plot(x_vals, y_vals, column, column_name, interval, interval_name, name, sub_path):
    plt.clf()
    sns.boxplot(x = x_vals, y = y_vals)
    plt.title(f"Box Plot: {name}")
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig(f"./images/{sub_path}/boxplot/{column}_{interval}.png")
    plt.close()



def histogram(data, column, column_name, index_name, sub_path):
    plt.clf()
    sns.histplot(data[column].values, kde = True)
    plt.title(f"Histogram: {index_name}")
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(f"./images/{sub_path}/histogram/{column}.png")
    plt.close()



def line_plot(x_vals, y_vals, column, column_name, interval, interval_name, index_name, sub_path):
    plt.clf()
    sns.lineplot(x = x_vals, y = y_vals)
    plt.title(f"Line Plot: {index_name}")
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig(f"./images/{sub_path}/lineplots/{column}_{interval}.png")
    plt.close()



def correlation_matrix(data, column, column_name, interval, sub_path):
    plt.clf()

    ax = sns.heatmap(data, annot = True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation = 30,
        horizontalalignment = "right"
    )

    plt.title(f"Heat Map: {column_name}")
    plt.savefig(f"./images/{sub_path}/matrices/{column}_{interval}.png")
    plt.close()



def heat_map(data, column, agg, interval, index_name, sub_path):
    plt.clf()
    sns.heatmap(data)
    plt.title(f"Heat Map: {index_name}")
    plt.savefig(f"./images/{sub_path}/heatmaps/{column}_{interval}_{agg}.png")
    plt.close()
