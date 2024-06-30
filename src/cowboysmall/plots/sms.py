
import matplotlib.pyplot as plt
import statsmodels.api as sm


from statsmodels.graphics.tsaplots import plot_acf



def qq_plot(values):
    sm.qqplot(values, line = '45', fit = True)
    plt.show()



def seasonal_plot(values, period):
    sm.tsa.seasonal_decompose(values, period = period).plot()
    plt.show()



def correlogram(values):
    plot_acf(values)
    plt.show()
