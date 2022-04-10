from typing import Dict, Iterable, Tuple
import lightgbm as lgb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""return distinct train and test sets"""
	return train_test_split(df, random_state = 0, test_size = 0.3)


def decompose(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
	"""break down data into features, labels and weights"""
	return df.drop(['y', 'w'], axis = 1), df.y, df.w


def ungroup(X: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
	"""expand all repeated columns"""
	return X.reindex(X.index.repeat(w))


def group(X: pd.DataFrame) -> pd.DataFrame:
	"""append weight to pandas dataframe by groupby all columns"""
	return X.groupby(list(X.columns)).size().to_frame('w').reset_index()


def get_feature_mappings(X: pd.DataFrame) -> Dict[str, Dict[str, int]]:
	"""map of value to pandas category"""
	return {
		f: {str(x): i
		    for i, x in enumerate(X[f].cat.categories, start = 1)}
		for f in X
	}


def etl(X: pd.DataFrame,
        y: pd.Series,
        w: pd.Series = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""transform the data"""
	X = X.assign(y = y)
	if w is not None:
		X_train, X_test = split(ungroup(X, w))
		return group(X_train), group(X_test)
	else:
		X['w'] = 1
		return split(X)


def predict(model: lgb.Booster,
            X: pd.DataFrame,
            regression: bool = False) -> pd.Series:
	"""prediction wrapper"""
	return model.predict(X) if regression else model.predict_proba(X)[:, 1]


def evaluate(X: pd.DataFrame,
             y: pd.Series,
             w: pd.Series,
             y_pred: pd.Series,
             regression: bool = False) -> Dict[str, any]:
	"""calculate evaluation metrics"""
	metrics = {'records': str(len(y)), 'weights': str(sum(w))}
	metrics['raw_accuracy'] = prediction_metrics(y, y_pred, weights = w)
	if not regression:
		metrics['log_loss'] = metrics.log_loss(y, y_pred, sample_weight = w)
		metrics['roc_auc_score'] = metrics.roc_auc_score(y, y_pred, sample_weight = w)
	return metrics


def prediction_metrics(var_y, var_y_pred, weights = None):
	if weights is None:
		weights = np.ones(len(var_y))
	avg = np.average(var_y, weights = weights)
	pred_avg = np.average(var_y_pred, weights = weights)
	bias2 = np.average(var_y - var_y_pred, weights = weights) ** 2
	var = np.average(var_y_pred ** 2, weights = weights) - np.average(var_y_pred, weights = weights) ** 2
	square_error = metrics.mean_squared_error(var_y, var_y_pred, weights)
	abs_err = metrics.mean_absolute_error(var_y, var_y_pred, weights)
	r2_score = metrics.r2_score(var_y, var_y_pred, weights)
	return dict(
		avg = avg, pred_avg = pred_avg, bias2 = bias2, var = var, square_error = square_error, abs_err = abs_err,
		abs_err_ratio = abs_err / avg, r2_score = r2_score
		)


def build_model(
		X: pd.DataFrame,
		y: pd.Series,
		w: pd.Series = None,
		categories: Iterable[str] = [],
		unencoded_categories: Iterable[str] = [],
		params: Dict[str, any] = {},
		regression: bool = False) -> Tuple[lgb.LGBMModel, Dict[str, any], pd.DataFrame, pd.DataFrame]:
	"""take the sql data and output model and metadata"""
	X[unencoded_categories] = X[unencoded_categories].astype('category')
	metadata = {
		'model_type': 'lightgbm',
		'features': list(X),
		'category_features': categories,
		'mappings': get_feature_mappings(X[unencoded_categories]),
	}

	train_df, test_df = etl(X, y, w)
	X, y, w = decompose(train_df)

	model = (lgb.LGBMRegressor if regression else lgb.LGBMClassifier)(random_state = 0, **params)
	model.fit(X, y, w, categorical_feature = categories)
	train_df['y_pred'] = predict(model, X, regression)

	metadata['train_accuracy'] = evaluate(X, y, w, train_df['y_pred'], regression)
	metadata['train_accuracy']['data_type'] = 'train'

	X, y, w = decompose(test_df)
	test_df['y_pred'] = predict(model, X, regression)
	metadata['test_accuracy'] = evaluate(X, y, w, test_df['y_pred'], regression)
	metadata['test_accuracy']['data_type'] = 'test'
	return model, metadata, train_df, test_df
