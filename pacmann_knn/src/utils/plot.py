import matplotlib.pyplot as plt

def plotting_predict_buy_no(x, y):
    """
    Goal is to find the best K value

    :param x:
    :param y:
    :return:
    """

    plt.title("Buy VS No K-Value")
    plt.ylabel("Accuracy")
    plt.xlabel("K Value")
    plt.plot(x, y)
    plt.savefig('foo.png', bbox_inches='tight')
    plt.show()

