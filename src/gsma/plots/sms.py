
import matplotlib.pyplot as plt
import statsmodels.api as sm


from statsmodels.graphics.tsaplots import plot_acf



def qq_plot(data, column, sub_path, figsize = (12, 9)):
    plt.clf()
    fig = sm.qqplot(data[column].values, line = '45', fit = True)
    fig.set_size_inches(figsize)
    fig.savefig(f"./images/{sub_path}/qqplots/{column}.png")
    plt.close()



def seasonal_plot(data, column, period, period_name, sub_path, figsize = (12, 9)):
    plt.clf()
    fig = sm.tsa.seasonal_decompose(data[column].values, period = period).plot()
    fig.set_size_inches(figsize)
    fig.savefig(f"./images/{sub_path}/seasonal/{column}_{period_name}.png")
    plt.close()



def correlogram(data, column, sub_path, figsize = (12, 9)):
    plt.clf()
    fig = plot_acf(data[column].values)
    fig.set_size_inches(figsize)
    fig.savefig(f"./images/{sub_path}/correlogram/{column}.png")
    plt.close()
