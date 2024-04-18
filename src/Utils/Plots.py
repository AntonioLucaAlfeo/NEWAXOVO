import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from random import randint, seed


class Plots:

    """
      This class implements some functions useful for plotting results.
    """

    @staticmethod
    def plot_feature_importance(labels, values, subject_id, sample_id):

        """
          This function creates a bar plot from the data passed in labels and values.

          | Pre-conditions: none.
          | Post-conditions: bar plot created from labels and values is shown.
          | Main output: none.

        :param labels: an array that specifies the labels of the bar plot.
        :type labels: ndarray.
        :param values: an array that specifies the values for the elements in *labels*.
        :type values: ndarray.
        :param sample_id: id of the sample to which data refer.
        :type sample_id: int (>0)
        :param subject_id: id of the subject to which data refer.
        :type subject_id: int (between 1 and 9).
        :return: none.
        :raise: none.
        """

        plt.figure(figsize=(15, 7))
        plt.bar(labels, values)
        plt.xlabel("Features")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Shap Values (abs)")
        title = "Shap Values (abs) of the Most Important Features (subject %d sample %d)" % (subject_id, sample_id)
        plt.title(title)

        base_path = "C:/Users/Luca/Desktop/NEWAXOVO/"
        filename = base_path + 'images/plots/feature_importance_subject_%d_sample_%d_%s.png' % (subject_id, sample_id,
                                                                            time.strftime("%Y%m%d-%H%M%S"))
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()

    @staticmethod
    def abs_shap(df_shap, df):

        """
          This function is taken from
          https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d.
          It generates a particular bar plot from the given data.

          | Pre-conditions: none.
          | Post-conditions: a new bar plot is generated from the given data.
          | Main output: none.

        :param df_shap: a dataframe containing shap values for each feature.
        :type df_shap: DataFrame.
        :param df: a dataframe containing a set of samples.
        :type df: DataFrame.
        :return: none.
        :raise: none.
        """

        # Make a copy of the input data
        shap_v = pd.DataFrame(df_shap)
        feature_list = df.columns
        shap_v.columns = feature_list
        df_v = df.copy().reset_index().drop('index', axis=1)

        # Determine the correlation in order to plot with different colors
        corr_list = list()
        for i in feature_list:
            b = np.corrcoef(shap_v[i], df_v[i])[1][0]
            corr_list.append(b)
        corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
        # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
        corr_df.columns = ['Variable', 'Corr']
        corr_df['Sign'] = np.where(corr_df['Corr'] > 0, 'red', 'blue')

        # Plot it
        shap_abs = np.abs(shap_v)
        k = pd.DataFrame(shap_abs.mean()).reset_index()
        k.columns = ['Variable', 'SHAP_abs']
        k2 = k.merge(corr_df, left_on='Variable', right_on='Variable', how='inner')
        k2 = k2.sort_values(by='SHAP_abs', ascending=True)
        colorlist = k2['Sign']
        ax = k2.plot.barh(x='Variable', y='SHAP_abs', color=colorlist, figsize=(5, 6), legend=False)
        ax.set_xlabel("SHAP Value (Red = Positive Impact)")

    @staticmethod
    def plot_grouped_bar_chart(labels, values, categories, x_label, y_label, xlim, title, display_error):

        """
          This function plots a grouped bar chart using the given parameters.

          | Pre-conditions: none.
          | Post-conditions: a new grouped bar chart is generated from the given data.
          | Main output: none.

        :param labels: labels that will be used into the bar chart. Each set of bars is associated with a label.
        :type labels: ndarray (n_labels,).
        :param values: values on which plot is created. Each value is in the form 'mean±std'.
        :type values: ndarray (n_categories, n_labels).
        :param categories: categories that will be used into the bar chart. Each set of bars is made up by
            n_categories bars.
        :type categories: ndarray (n_categories,).
        :param x_label: name associated to the x axis.
        :type x_label: str.
        :param y_label: name associated to the y axis.
        :type y_label: str.
        :param xlim: lower and upper bound of the x axis.
        :type xlim: ndarray (2,).
        :param title: title of the plot.
        :type title: str.
        :param display_error: True if error lines are needed.
        :type display_error: boolean.
        :return: none.
        :raise: none.
        """

        means = np.empty((values.shape[0], values.shape[1]))
        stds = np.empty((values.shape[0], values.shape[1]))

        for row_index in np.arange(values.shape[0]):

            for column_index in np.arange(values.shape[1]):

                mean = values[row_index, column_index].split("±")[0].replace(" ", "")
                std = values[row_index, column_index].split("±")[1].replace(" ", "")

                means[row_index, column_index] = float(mean)
                stds[row_index, column_index] = float(std)

        means = means.transpose()
        stds = stds.transpose()

        mean_dataframe = pd.DataFrame(data=means, columns=categories, index=labels)
        std_dataframe = pd.DataFrame(data=stds, columns=categories, index=labels)

        fig, ax = plt.subplots()

        if display_error:
            ax = mean_dataframe.plot.barh(ax=ax, xlim=xlim, title=title, xerr=std_dataframe)
        else:
            ax = mean_dataframe.plot.barh(ax=ax, xlim=xlim, title=title)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.show()
        plt.clf()

    @staticmethod
    def plot_bar_chart(values, categories, x_label, xlim, title, display_error):

        """
          This function plots a bar chart using the given parameters.

          | Pre-conditions: none.
          | Post-conditions: a new bar chart is generated from the given data.
          | Main output: none.

        :param values: values on which plot is created. Each value is in the form 'mean±std'.
        :type values: ndarray (n_categories,).
        :param categories: categories that will be used into the bar chart. A category will represent a label on the
            axis.
        :type categories: ndarray (n_categories,).
        :param x_label: name associated to the x axis.
        :type x_label: str.
        :param xlim: lower and upper bound of the x axis.
        :type xlim: ndarray (2,).
        :param title: title of the plot.
        :type title: str.
        :param display_error: True if error lines are needed.
        :type display_error: boolean.
        :return: none.
        :raise: none.
        """

        means = np.empty(values.shape[0])
        stds = np.empty(values.shape[0])

        for row_index in np.arange(values.shape[0]):

            mean = values[row_index].split("±")[0].replace(" ", "")
            std = values[row_index].split("±")[1].replace(" ", "")

            means[row_index] = float(mean)
            stds[row_index] = float(std)

        mean_dataframe = pd.DataFrame({'lab': categories, 'val': means, "err": stds})
        std_dataframe = pd.DataFrame({'lab': categories, 'val': stds})
        print(std_dataframe)
        fig, ax = plt.subplots()

        seed(19)

        if display_error:
            ax = mean_dataframe.plot.barh(x='lab', y='val', ax=ax, xlim=xlim, title=title, xerr='err',
                                          color=['#%06X' % randint(0, 0xFFFFFF) for _ in np.arange(means.shape[0])],
                                          legend=False, capsize=4)
        else:
            ax = mean_dataframe.plot.barh(x='lab', y='val', ax=ax, xlim=xlim, title=title,
                                          color=['#%06X' % randint(0, 0xFFFFFF) for _ in np.arange(means.shape[0])],
                                          legend=False)

        ax.set_xlabel(x_label)
        plt.show()
        plt.clf()

    @staticmethod
    def plot_confusion_matrix(cm, cms, classes, filepath, title, cmap=plt.cm.Blues):

        """
          This function prints and plots the confusion matrix, given matrices of means and stds. Taken from
          https://stackoverflow.com/questions/59319533/plot-a-confusion-matrix-in-python-using-a-dataframe-of-strings.

          | Pre-conditions: none.
          | Post-conditions: a new confusion matrix is generated from the given data.
          | Main output: none.

        :param cm: matrix of means.
        :type cm: ndarray (n_classes,n_classes).
        :param cms: matrix of std.
        :type cms: ndarray (n_classes,n_classes).
        :param classes: labels of classes.
        :type classes: ndarray (n_classes,)
        :param filepath: filepath in which confusion matrix image is stored.
        :type filepath: str.
        :param title: title of the plot.
        :type title: str.
        :param cmap: color map.
        :type cmap: str or Colormap
        :return: none.
        :raise: none.
        """

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        thresh = cm.max() / 2.

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, '{0:.4f}'.format(cm[i, j]) + '\n$\pm$' + '{0:.4f}'.format(cms[i, j]),
                         horizontalalignment="center",
                         verticalalignment="center", fontsize=9,
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)

        plt.savefig(filepath, bbox_inches='tight')
        plt.clf()
