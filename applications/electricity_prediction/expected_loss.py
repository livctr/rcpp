from typing import List, Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import copy
import re
from rcpp.width_calculator import CLTWidth
from rcpp.risk_measure import MeanRiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.loss_simulator import LossSimulator
from rcpp.main import run_experiment
from utils_xg import evaluate, gen_prompt, tree_to_code, get_cart, load_model, use_api


def rule_template(rule):
    variables = "x1"
    for i in range(1, len(xtrain[0])):
        variables += ", x{:.0f}".format(i+1)
    target_variable = "x{:.0f}".format(len(xtrain[0])+1)
    text = f'''
import numpy as np

def rule(data):
    [{variables}] = data
    {rule}
    return {target_variable}[0]
    '''
    return text

class ElectricityShiftSimulator(PerformativitySimulator):

    def __init__(self, model, tokenizer, classifier_model, org_feat_num, seed=42):
        """
        Arguments:
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        # TODO set params
        self.model = model
        self.tokenizer = tokenizer
        self.classifier_model = classifier_model
        self.seed = seed
        self.org_feat_num = org_feat_num
        self.train_acc_list = []
        self.score_list = []
        self.test_acc_list = []
        self.r_list = []
        self.dt_list = []
        self.iteration = 0
        self.pattern = r"x{}\s*=\s*\[.*?\]".format( + 1)
        self.train_initia_predictor()

    def train_initia_predictor(self):
        # Train initial predictor
        r0 = "x{:.0f} = [x{:.0f} * x{:.0f}]".format(self.org_feat_num + 1, 1, 2)
        rule_text = rule_template(r0)
        exec(rule_text, globals())
        new_col = [(rule(xtrain[i])) for i in range(len(xtrain))]
        new_col += [(rule(xval[i])) for i in range(len(xval))]
        train_acc, val_acc, self.classifier_model = evaluate(new_col, xtrain, ytrain, xval, yval, self.classifier_model)
        best_CART = get_cart(new_col, xtrain, ytrain, xval, yval, self.seed) # Train CART
        dt0 = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, self.org_feat_num + 2)]) # Tree to Text
        # append
        self.r_list.append(r0)
        self.score_list.append(val_acc)
        self.train_acc_list.append(train_acc)
        self.dt_list.append(dt0)

    @property
    def add_constraint(self):
        return self.iteration >= 40

    def shift(self, Z: List[np.ndarray]) -> List[np.ndarray]:
        xtrain, ytrain = Z[0], Z[1]
        # Split to train, val
        n_total = len(xtrain)
        n_20 = int(0.2 * n_total)
        indices = np.random.permutation(n_total)
        xval = xtrain[indices[:n_20]]
        yval = ytrain[indices[:n_20]]
        xtrain = xtrain[indices[n_20:]]
        ytrain = ytrain[indices[n_20:]]

        prompt = gen_prompt(self.r_list, self.dt_list, self.score_list, self.org_feat_num+1, self.add_constraint)
        all_status = 0
        while all_status == 0:
            answer_temp1 = use_api(prompt, self.model, self.tokenizer, 1.0)

            for num_iter in range(len(answer_temp1)):                
                match = re.search(self.pattern, answer_temp1[num_iter], re.DOTALL)
                
                if match:
                    try:
                        extracted_text = match.group()
                        rule_text = rule_template(extracted_text)
                        exec(rule_text, globals())
                        new_col = [(rule(xtrain[i])) for i in range(len(xtrain))]
                        new_col += [(rule(xval[i])) for i in range(len(xval))]
                        train_acc, val_acc, self.classifier_model = evaluate(new_col, xtrain, ytrain, xval, yval, self.classifier_model)
                        best_CART = get_cart(new_col, xtrain, ytrain, xval, yval, self.seed) # Train CART
                        dt = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, len(xtrain[0]) + 2)])                  
                        self.r_list.append(extracted_text)
                        self.score_list.append(val_acc)
                        self.train_acc_list.append(train_acc)
                        self.dt_list.append(dt)    
                        all_status = 1
                        
                        for value in new_col:
                            if value == np.inf:
                                all_status = 0
                            elif value == -np.inf:
                                all_status = 0
                            elif np.isnan(value):
                                all_status = 0
                    except Exception as e:
                        with open("err.log", "a") as f:
                            f.write(str(e))      
                            f.write("\n")
                else:
                    pass

        self.iteration += 1

        # add_column
        gen_c = np.array(gen_c).reshape(-1,1)
        enc = MinMaxScaler()
        enc = enc.fit(gen_c)
        gen_c = enc.transform(gen_c)
        new_train = np.concatenate([xtrain, gen_c], axis = -1)
        
        return (new_train, ytrain)

    def simulate_shift(self,
                       Z_base: Union[List[np.ndarray], None],
                       Z_prev: Union[List[np.ndarray], None],
                       lambda_: float,
                       gamma: float) -> List[np.ndarray]:
        # TODO ignore Z_base, ignore lambda_, ignore gamma; shift `Z_prev`
        assert len(Z_prev) == 2
        X_prev, Y_prev = Z_prev
        X_shifted, Y_shifted = self.shift(Z_prev)
        return [X_shifted, Y_shifted]


class ElectricityLossSimulator(LossSimulator):

    def __init__(self, classifier_model):
        self.classifier_model = classifier_model

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True) -> np.ndarray:
        assert len(Z) == 2
        X, Y = Z
        Y_hat = self.classifier_model.predict_proba(X)[:, 1]

        loss = np.where(
            (0.5 - lambda_ <= Y_hat) & (Y_hat <= 0.5 + lambda_),  # abstain from prediction
            0,
            np.where(
                Y_hat < 0.5 - lambda_,
                np.where(Y == 1, 1, 0),  # Predict 0, loss if Y == 1
                np.where(Y == 0, 1, 0)  # Predict 1, loss if Y == 0
            )
        )
        return loss


class Args:
    def __init__(self):
        self.alpha = 0.3         # risk control level
        self.tightness = 0.08    # tightness parameter, may throw error if too low
        self.delta = 0.1         # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 0.5
        self.gamma = 0.1         # parameter in the Wasserstein distance formulation, for simulating distribution shift


if __name__ == "__main__":

    np.random.seed(42)

    # TODO: Load data
    xtrain = np.load(f'xtrain.npy')
    xval = np.load(f'xval.npy')
    xtest = np.load(f'xtest.npy')
    ytrain = np.load(f'ytrain.npy')
    yval = np.load(f'yval.npy')
    ytest = np.load(f'ytest.npy')
    X_all = np.concatenate([xval, xtest], axis=0)
    Y_all = np.concatenate([yval, ytest], axis=0)

    llm_model, tokenizer = load_model('meta-llama/Llama-3.1-8B-Instruct', None)  # model that does shifting

    classifier_model = LogisticRegression(fit_intercept=False)

    # Run experiment
    args = Args()
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = ElectricityShiftSimulator(model=llm_model, tokenizer=tokenizer, classifier_model=classifier_model, org_feat_num=len(xtrain[0]))
    loss_simulator = ElectricityLossSimulator(classifier_model)
    run_experiment(
        Z=[X_all, Y_all],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        gammas=[1.0],
        save_dir=f"./applications/electricity_prediction/figures/expected_loss/",
        num_iters=10  # TODO: set to 1000 for real experiments, 10 for testing/debugging
    )
