
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf



def plot_setup(figsize = (8, 6), style = "ggplot", sns_style = "darkgrid", sns_context = "paper"):
    plt.figure(figsize = figsize)
    plt.style.use(style)

    sns.set_style(sns_style)
    sns.set_context(sns_context)



def scatter_plot(data, column, column_name, index_name):
    plt.clf()
    plt.scatter(data.index, data[column].values)
    plt.title(f"Scatter Plot: {index_name}")
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.savefig(f"./images/indices/scatter/{column}.png")
    plt.close()




def histogram(data, column, column_name, index_name):
    plt.clf()
    sns.histplot(data[column].values, kde = True)
    plt.title(f"Histogram: {index_name}")
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig(f"./images/indices/histogram/{column}.png")
    plt.close()


def box_plot(x_vals, y_vals, column, column_name, interval, interval_name, index_name):
    plt.clf()
    sns.boxplot(x = x_vals, y = y_vals)
    plt.title(f"Box Plot: {index_name}")
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig(f"./images/indices/boxplot/{column}_{interval}.png")
    plt.close()


def line_plot(x_vals, y_vals, column, column_name, interval, interval_name, index_name):
    plt.clf()
    sns.lineplot(x = x_vals, y = y_vals)
    plt.title(f"Line Plot: {index_name}")
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig(f"./images/indices/lineplots/{column}_{interval}.png")
    plt.close()




def qq_plot(data, column):
    plt.clf()
    fig = sm.qqplot(data[column].values, line = '45', fit = True)
    fig.savefig(f"./images/indices/qqplots/{column}.png")
    plt.close()


def seasonal_plot(data, column, period, period_name):
    plt.clf()
    fig = sm.tsa.seasonal_decompose(data[column].values, period = period).plot()
    fig.savefig(f"./images/indices/seasonal/{column}_{period_name}.png")
    plt.close()




def correlogram(data, column):
    plt.clf()
    fig = plot_acf(data[column].values)
    fig.savefig(f"./images/indices/correlogram/{column}.png")
    plt.close()
