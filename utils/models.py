from sklearn import model_selection

class Model:
    
    NUM_FOLDS = 5

    def __init__(self, data_x, data_y, model, metric, randomSeed=42):
        self.randomSeed = randomSeed
        self.X = data_x
        self.y = data_y
        self.metric = metric
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS)
        self.model = model

    def __len__(self):
        return self.X.shape[1]

    def getMeanAccuracy(self, zeroOneList):
        # drop unselected features
        zeroIndices = [i for i, n in enumerate(zeroOneList) if n == 0]
        currentX = self.X.drop(self.X.columns[zeroIndices], axis=1)

        # k-fold validation 
        cv_results = model_selection.cross_val_score(
            self.model, 
            currentX, 
            self.y, 
            cv=self.kfold, 
            scoring=self.metric
            )

        return cv_results.mean()
