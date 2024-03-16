
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
