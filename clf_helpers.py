import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, TransformerMixin


class BaseClf(object):
    def __init__(self, dataframe, classifier, features, target, scale_x=True):
        self.dataframe_ = dataframe.copy()
        self.features_ = features
        self.X_ = self.to_input_array(self.dataframe_, self.features_, scale_x = scale_x)
        self.classifier_ = classifier
        self.target_ = target
        self.y_ = dataframe[self.target_].values
        self.classes_ = np.unique(self.y_)

    def to_input_array(self, dataframe, features, scale_x = True):
        """return scaled (default) or unscaled array of inputs"""
        return StandardScaler().fit_transform(dataframe[features].values) if scale_x == True \
            else dataframe[features].values

    def cv_predict(self, x, y = None, method = 'predict'):
        """cross validated predictions (stratifiedKfold, 3 folds)"""
        preds = cross_val_predict(self.classifier_, x, y, method = method)
        return preds

    def conf_mtx(self, y, preds, classes):
        """compute confusion matrix"""
        cm = pd.crosstab(index = y, columns = preds, margins = True)
        cm.rename({'All': 'Actual'}, axis = 'columns', inplace=True)
        cm.rename({'All': 'Predicted'}, axis = 'index', inplace = True)
        return cm

    def rank_features(self, features, classes):
        """return series of features and ranks if clf has coef/feature importance attribs"""
        if hasattr(self.classifier_, 'feature_importances_'):
            ranks = pd.Series(self.classifier_.feature_importances_, index = features).sort_values()
            return ranks
        elif hasattr(self.classifier_, 'coef_'):
            if len(classes) > 2:
                ranks = pd.DataFrame(self.classifier_.coef_,
                                     index = classes,
                                     columns = features)
            else:
                ranks = pd.Series(self.classifier_.coef_.flatten(),
                                  index = features).sort_values()
            return ranks
        else:
            return "n/a"

    def Predict(self, dataframe, features, scale_x=True):
        """non cv predictions"""
        input_arr = self.to_input_array(dataframe, features, scale_x=scale_x)
        preds = self.classifier_.predict(input_arr)
        return preds


    def ova_pr(self, cm, pos_label):
        """for multi-class: calculate one versus all precision/recall from cv preds in confusion matrix"""
        precision = cm.loc[pos_label, pos_label] / cm.loc['Predicted', pos_label]
        recall = cm.loc[pos_label, pos_label] / cm.loc[pos_label, 'Actual']
        return precision, recall


    def plot_cm(self, ax, cm, scores = None):
        """plots heatmap of confusion matrix with diagonal color set to 0, annot with percents"""
        norm_m = cm.iloc[:-1, :-1].values/cm.iloc[:-1,-1].values.reshape(-1,1)
        norm_m2 = np.copy(norm_m)
        np.fill_diagonal(norm_m2, 0)
        if scores is not None:
            f1_mean = scores.iloc[:, 0].mean()
            f1_std = scores.iloc[:, 0].std()
            ax.set_title('confusion matrix; \nCV mean f1 score: {:.2f} +/- {:.2f}'.format(f1_mean,f1_std))
        sns.heatmap(norm_m2, ax = ax, annot = norm_m, cbar = False)
        return ax

    def plot_features(self, ax, feature_ranks):
        """plots feature importance"""
        if type(feature_ranks) != str:
            feature_ranks.plot.barh(ax = ax, fontsize = 10)
            ax.set_title('feature importance')
            return ax
        else:
            "n/a"

    def plot_scatter(self, ax, x_label, y_label, preds):
        """plot scatter of columns in original dataframe colored by class"""
        colors = ['#AEE1AE', 'r', 'navy']
        self.dataframe_['class_preds'] = preds

        for cls, color in zip(self.classes_, colors):
            ax.scatter(self.dataframe_.loc[self.dataframe_.class_preds == cls, x_label],
                       self.dataframe_.loc[self.dataframe_.class_preds == cls, y_label],
                       color = color, alpha = .5, label = cls)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.set_title('{0} vs {1} by predicted class'.format(x_label, y_label))
        return ax

    #def grid_search(self, param_grid, x, y...)
    #GridSearchCV(self.classifier_, param_grid, x, y, )

class ClfS(BaseClf):
    """supervised classifier"""
    def __init__(self, dataframe, classifier, features, target, scale_x=True):
        super().__init__(dataframe, classifier, features, target, scale_x)
        self.preds_ = self.cv_predict(x = self.X_, y = self.y_)
        self.classifier_ = self.classifier_.fit(self.X_, self.y_)
        self.cm_ = self.conf_mtx(self.y_, self.preds_, self.classes_)
        self.scores_ = self.cv_score(self.X_, self.y_, self.classes_)
        self.pr_curve_ = self.pr_curve(self.X_, self.y_, self.classes_)
        self.feature_ranks_ = self.rank_features(self.features_, self.classes_)

    def pr_curve(self, x, y, classes):
        """cross validated precision/recall curve across varying thresholds"""
        # predict class probabilities based on classifiers method
        if hasattr(self.classifier_, 'decision_function'):
            method = 'decision_function'
        else:
            method = 'predict_proba'
        proba = self.cv_predict(x, y, method)


        # multi class
        if len(classes) > 2:
            # change y to multi-class form
            new_y = label_binarize(y, classes = classes.tolist())

            # For each class
            precision = dict()
            recall = dict()
            average_precision = dict()

            for i in range(len(classes)):
                precision[i], recall[i], _ = precision_recall_curve(new_y[:, i],proba[:, i])
                average_precision[i] = average_precision_score(new_y[:, i],proba[:, i])

            pr = {'precision': precision,
                  'recall': recall,
                  'avg_precision': average_precision}

        # binary
        else:
            if method == 'predict_proba': # proba returns probabilities for all classes
                proba = proba[:, 1] # isolate positive class probabilities
            average_precision = average_precision_score(y, proba)
            precision, recall, _ = precision_recall_curve(y, proba)

            pr = {'precision': precision,
                  'recall': recall,
                  'avg_precision': average_precision}

        return pr

    def cv_score(self, x, y, classes):
        """cross validated f1, recall, precision scores (macro for multi-class) (stratifiedKfold, 3 folds)"""
        if len(classes) == 2:
            scoring = ['f1', 'recall', 'precision']
        else:
            scoring = ['f1_macro', 'recall_macro', 'precision_macro']
        scores = cross_validate(self.classifier_, x, y, scoring = scoring, return_train_score = False)
        scores = {k: v for k, v in scores.items() if 'time' not in k}
        return pd.DataFrame(scores)

    def plot_rp(self, ax):
        """plot recall precision curve"""
        # multi class
        if len(self.classes_) > 2:
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = ax.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                ax.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

            lines.append(l)
            labels.append('iso-f1 curves')

            colors = ['navy', 'red', 'turquoise']

            for i, color in zip(range(self.classes_.shape[0]), colors):
                l, = ax.plot(self.pr_curve_['recall'][i], self.pr_curve_['precision'][i], color=color, lw=2)
                lines.append(l)
                labels.append('Precision-recall for class {0} (area = {1:0.2f})'.format(i, self.pr_curve_['avg_precision'][i]))

            ax.legend(lines, labels, loc = (0, -.5))
            ax.set_title('Precision-Recall')

        else:
            ax.plot(self.pr_curve_['recall'], self.pr_curve_['precision'], color='b', lw=2)
            ax.set_title('Precision-Recall AP={0:0.2f}'.format(self.pr_curve_['avg_precision']))

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        return ax



class ClfRFE(ClfS):
    """perform RFECV using supervised learning classifier that has feature importance/coef attrbs"""

    def __init__(self, dataframe, classifier, features, target, fs_scoring, scale_x=True):
        super().__init__(dataframe, classifier, features, target, scale_x)
        self.rfecv_ = self.Fit(classifier, fs_scoring, self.X_, self.y_)
        self.classifier_ = self.rfecv_.estimator_ # classifier is set to estimator fit on reduced data
        self.new_X_ = self.rfecv_.transform(self.X_)
        self.best_features_ = list(map(lambda x: x[0],
                                       filter(lambda x: x[1] == True,
                                              zip(self.features_, self.rfecv_.support_))
                                       ))

        self.preds_ = self.cv_predict(x = self.new_X_, y = self.y_)
        self.cm_ = self.conf_mtx(self.y_, self.preds_, self.classes_)
        self.scores_ = self.cv_score(self.new_X_, self.y_, self.classes_)
        self.pr_curve_ = self.pr_curve(self.new_X_, self.y_, self.classes_)
        self.feature_ranks_ = self.rank_features(self.best_features_, self.classes_)

    def Fit(self, classifier, scoring, x, y):
        """fits RFECV based on scoring method"""
        rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring= scoring)
        rfecv.fit(x, y)
        return rfecv

    def plot_fs(self, ax):
        """plot cv scores"""
        ax.plot(range(1, len(self.rfecv_.grid_scores_) + 1), self.rfecv_.grid_scores_)
        ax.set_title('Opt # Features: ' + str(self.rfecv_.n_features_))
        return ax

class ClfU(BaseClf):
    def __init__(self, dataframe, classifier, features, target, scale_x=True):
        super().__init__(dataframe, classifier, features, target, scale_x)
        self.classifier_ = classifier.fit(self.X_)
        self.preds_ = self.cv_predict(x = self.X_)
        self.cm_ = self.conf_mtx(self.y_, self.preds_, self.classes_)
        self.scores_ = self.cv_score(self.X_, self.y_)
        self.feature_ranks_ = self.rank_features(self.features_, self.classes_)

    def cv_predict(self, x, y = None, method = 'predict'):
        preds = super().cv_predict(x, y, method)
        if method == 'predict':
            preds = np.where(preds == 1, 0, 1)
        return preds

    def Predict(self, dataframe, features, scale_x=True):
        preds = super().Predict(dataframe, features, scale_x)
        preds = np.where(preds == 1, 0, 1)
        return preds

    def cv_score(self, x, y):
        skf = StratifiedKFold(n_splits = 3, random_state = 42)
        scores = pd.DataFrame(columns = ['f1','precision', 'recall'])
        i = 0
        for train_idx, test_idx in skf.split(x, y):
            x_tr, x_ts = x[train_idx], x[test_idx]
            y_tr, y_ts = y[train_idx], y[test_idx]
            preds_ts = self.classifier_.fit(x_tr).predict(x_ts)
            preds_ts = np.where(preds_ts == 1, 0, 1)
            f1 = f1_score(y_ts, preds_ts)
            precision = precision_score(y_ts, preds_ts)
            recall = recall_score(y_ts, preds_ts)
            scores.loc[i] = [f1, precision, recall]
            i += 1
        return scores

    def rank_features(self, features, classes):
        ranks = super().rank_features(features, classes)
        if hasattr(self.classifier_, 'estimators_'):
            if hasattr(self.classifier_.estimators_[0], 'feature_importances_'):
                fi_list = list(map(lambda x: x.feature_importances_, self.classifier_.estimators_))
            elif hasattr(self.classifier_.estimators_[0], 'coef_'):
                fi_list = list(map(lambda x: x.coef_, self.classifier_.estimators_))
            else:
                fi_list = 'na'
            meta_ranks = pd.melt(pd.DataFrame(fi_list, columns = features)) if type(fi_list) == list else 'na'
        return meta_ranks

    def plot_meta_features(self, ax, feature_ranks):
        sns.pointplot(y='variable',
                      x = 'value',
                      data = feature_ranks,
                      join = False,
                      orient = 'h',
                      ax = ax,
                      order = feature_ranks.groupby('variable').value.mean().sort_values(ascending = False).index)


class DataFrameSelectorLite(BaseEstimator, TransformerMixin):
    """Selects columns given in attribute_names list"""
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """Selects list of columns corresponding to set_selector key from attribute_names dict"""
    def __init__(self, attribute_names, set_selector):
        self.attribute_names = attribute_names
        self.set_selector = set_selector
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names[self.set_selector]].values
    
class ArraySelector(BaseEstimator, TransformerMixin):
    """Selects list of columns corresponding to set_selector key from attribute_names dict"""
    def __init__(self, var_dict, set_selector, feature_names):
        self.var_dict = var_dict
        self.set_selector = set_selector
        self.feature_names = feature_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[:, [self.feature_names.index(var) for var in self.var_dict[self.set_selector]]]
    def get_feature_names(self):
        return self.var_dict[self.set_selector]
