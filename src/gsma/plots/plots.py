
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf


def scatter_plot(data, column, column_name, index):
    plt.clf()
    plt.scatter(data.index, data[column])
    plt.title('Scatter Plot: {}'.format(index))
    plt.xlabel('Date')
    plt.ylabel(column_name)
    plt.savefig("./images/indices/scatter/{}.png".format(column))
    plt.close()


def box_plot(data, column, column_name, index):
    plt.clf()
    sns.boxplot(x = 'Y', y = data[column], data = data)
    plt.title('Box Plot: {}'.format(index))
    plt.xlabel('Year')
    plt.ylabel(column_name)
    plt.savefig("./images/indices/boxplot/{}.png".format(column))
    plt.close()


def qq_plot(data, column):
    plt.clf()
    fig = sm.qqplot(data[column], line = '45', fit = True)
    fig.savefig("./images/indices/qqplots/{}.png".format(column))
    plt.close()


def histogram(data, column, column_name, index):
    plt.clf()
    sns.histplot(data[column], kde = True)
    plt.title('Histogram: {}'.format(index))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig("./images/indices/histogram/{}.png".format(column))
    plt.close()


def line_plot(x_vals, data, column, column_name, interval, interval_name, index):
    plt.clf()
    sns.lineplot(x = x_vals, y = data[column])
    plt.title('Line Plot: {}'.format(index))
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig("./images/indices/lineplots/{}_{}.png".format(column, interval))
    plt.close()


def seasonal_plot(data, column, p_name, p_value):
    plt.clf()
    fig = sm.tsa.seasonal_decompose(data[column].values, period = p_value).plot()
    fig.savefig("./images/indices/seasonal/{}_{}.png".format(column, p_name))
    plt.close()


def correlogram(data, column):
    plt.clf()
    fig = plot_acf(data[column].values)
    fig.savefig("./images/indices/correlogram/{}.png".format(column))
    plt.close()
