import os
import re
from copy import deepcopy
from typing import List, Union, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from rcpp.loss_simulator import LossSimulator, ZeroOneLossSimulator
from rcpp.main import (
    plot_final_loss_vs_iteration,
    plot_lambda_vs_iteration,
    plot_loss_vs_iteration,
    run_trajectory,
    Trajectory
)
from rcpp.performativity_simulator import PerformativitySimulator
from rcpp.risk_measure import MeanRiskMeasure, RiskMeasure
from rcpp.width_calculator import CLTWidth, WidthCalculator
from .utils_xg import evaluate, gen_prompt, get_cart, load_model, tree_to_code, use_api



def rule_template(rule, org_feat_num):
    variables = "x1"
    for i in range(1, org_feat_num):
        variables += ", x{:.0f}".format(i+1)
    target_variable = "x{:.0f}".format(org_feat_num+1)
    text = f'''
import numpy as np

def rule(data):
    [{variables}] = data
    {rule}
    return {target_variable}[0]
    '''
    return text

class ElectricityShiftSimulator(PerformativitySimulator):

    def __init__(self, model, tokenizer, classifier_model, data, seed=42):
        """
        Arguments:
        - M: bound on the base distribution PDF of the predictions Y_hat (not the loss)
        - beta: the beta in beta-CVaR, e.g., 0.9 for 90%
        """
        self.model = model
        self.tokenizer = tokenizer
        self.classifier_model = classifier_model
        self.seed = seed
        self.xtrain, self.ytrain, self.xval, self.yval = data
        self.org_feat_num = len(self.xtrain[0])
        self.pattern = r"x{}\s*=\s*\[.*?\]".format(self.org_feat_num + 1)

    def reset(self):
        self.train_acc_list = []
        self.score_list = []
        self.test_acc_list = []
        self.r_list = []
        self.dt_list = []
        self.iteration = 0
        # Train initial predictor
        r0 = "x{:.0f} = [x{:.0f} * x{:.0f}]".format(self.org_feat_num + 1, 1, 2)
        rule_text = rule_template(r0, self.org_feat_num)
        exec(rule_text, globals())
        new_col = [(rule(self.xtrain[i])) for i in range(len(self.xtrain))]
        new_col += [(rule(self.xval[i])) for i in range(len(self.xval))]
        train_acc, val_acc, self.classifier_model = evaluate(new_col, self.xtrain, self.ytrain, self.xval, self.yval, self.classifier_model)
        best_CART = get_cart(new_col, self.xtrain, self.ytrain, self.xval, self.yval, self.seed) # Train CART
        dt0 = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, self.org_feat_num + 2)]) # Tree to Text
        # append
        self.r_list.append(r0)
        self.score_list.append(val_acc)
        self.train_acc_list.append(train_acc)
        self.dt_list.append(dt0)

    @property
    def add_constraint(self):
        return self.iteration >= 40
        

    def _simulate_shift(self, Z: List[np.ndarray]) -> List[np.ndarray]:
        # Shift on the test data. Use train and val data to train model
        xtest, ytest = Z[0], Z[1]

        prompt = gen_prompt(self.r_list, self.dt_list, self.score_list, self.org_feat_num+1, self.add_constraint)
        while True:
            answer_temp1 = use_api(prompt, self.model, self.tokenizer, 1.0)
            answer_temp1 = answer_temp1[0]
            match = re.search(self.pattern, answer_temp1, re.DOTALL)
            if match:
                try:
                    extracted_text = match.group()
                    rule_text = rule_template(extracted_text, self.org_feat_num)
                    exec(rule_text, globals())
                    new_col = [(rule(self.xtrain[i])) for i in range(len(self.xtrain))]
                    new_col += [(rule(self.xval[i])) for i in range(len(self.xval))]
                    train_acc, val_acc, self.classifier_model = evaluate(new_col, self.xtrain, self.ytrain, self.xval, self.yval, self.classifier_model)
                    best_CART = get_cart(new_col, self.xtrain, self.ytrain, self.xval, self.yval, self.seed) # Train CART
                    dt = tree_to_code(best_CART, ['x{}'.format(i) for i in range(1, len(self.xtrain[0]) + 2)])                  
                    self.r_list.append(extracted_text)
                    self.score_list.append(val_acc)
                    self.train_acc_list.append(train_acc)
                    self.dt_list.append(dt)

                    new_col = np.array(new_col).reshape(-1,1)
                    if not np.isfinite(new_col).all():
                        continue

                    # Change test data
                    if len(xtest[0]) == self.org_feat_num + 1:
                        xtest = np.delete(xtest, -1, axis=1)
                    new_col = [(rule(xtest[i])) for i in range(len(xtest))]
                    new_col = np.array(new_col).reshape(-1,1)
                    if not np.isfinite(new_col).all():
                        continue
                    enc = MinMaxScaler()
                    enc = enc.fit(new_col)
                    new_col = enc.transform(new_col)
                    new_xtest = np.concatenate([xtest, new_col], axis = -1)
                    self.iteration += 1
                    return [new_xtest, ytest]

                except Exception as e:
                    with open("err.log", "a") as f:
                        f.write(str(e))      
                        f.write("\n")
            else:
                pass

    def simulate_shift(self,
                       Z_base: Union[List[np.ndarray], None],
                       Z_prev: Union[List[np.ndarray], None],
                       lambda_: float,
                       gamma: float) -> List[np.ndarray]:
        assert len(Z_prev) == 2  # Z_prev = [X_prev, Y_prev]
        return self._simulate_shift(Z_prev)


class ElectricityLossSimulator(LossSimulator):

    def __init__(self, classifier_model):
        self.classifier_model = classifier_model

    def calc_loss(self, Z: List[np.ndarray], lambda_: float, do_new_sample: bool = True) -> np.ndarray:
        assert len(Z) == 2
        X, Y = Z
        Y_hat = self.classifier_model.predict_proba(X)[:, 1]

        loss = np.where(
            (0.5 - lambda_ / 2. <= Y_hat) & (Y_hat <= 0.5 + lambda_ / 2.),  # abstain from prediction
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
        self.alpha = 0.25         # risk control level
        self.tightness = 0.08    # tightness parameter, may throw error if too low
        self.delta = 0.1         # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_max = 1.0    # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 0.1         # placeholder


def run_electricity_experiment(
    Z,
    width_calculator: WidthCalculator,
    risk_measure: RiskMeasure,
    performativity_simulator: PerformativitySimulator,
    loss_simulator: LossSimulator,
    args,
    save_dir: str = "./figures",
    num_iters: int = 1000,
):
    """
    Run the RCPP algorithm and plot the results.
    """

    def cut(data):
        tot = len(data[0])
        assert tot > args.N
        idx = np.random.choice(tot, size=args.N, replace=False)
        data_cal = [d[idx] for d in data]
        data_test = [d[~idx] for d in data]
        return data_cal, data_test

    # # Plot 1: lambda trajectory
    # Z_cal, Z_test = cut(Z)
    # trajectory = run_trajectory(
    #     Z_cal,
    #     Z_test,
    #     width_calculator,
    #     risk_measure,
    #     performativity_simulator,
    #     loss_simulator,
    #     args
    # )
    # plot_lambda_vs_iteration([0], [trajectory], args, save_dir=save_dir)

    # Control risk
    Z_cal, Z_test = cut(Z)
    trajectories_controlled = []
    for _ in tqdm(range(num_iters), desc="Running trials"):
        trajectory = run_trajectory(
            Z_cal,
            Z_test,
            width_calculator,
            risk_measure,
            performativity_simulator,
            loss_simulator,
            args
        )
        trajectories_controlled.append(trajectory)

    trajectories_uncontrolled = []
    for _ in tqdm(range(num_iters), desc="Running trials"):
        Z_cal, Z_test = cut(Z)
        trajectory = run_trajectory(
            Z_cal,
            Z_test,
            width_calculator,
            risk_measure,
            performativity_simulator,
            loss_simulator,
            args,
            control_risk=False,
            num_iters=10
        )
        trajectories_uncontrolled.append(trajectory)
    
    controlled_save_dir = os.path.join(save_dir, "controlled")
    os.makedirs(controlled_save_dir, exist_ok=True)
    plot_loss_vs_iteration(trajectories_controlled, args, save_dir=controlled_save_dir)
    plot_final_loss_vs_iteration(trajectories_controlled, args, save_dir=controlled_save_dir)
    uncontrolled_save_dir = os.path.join(save_dir, "uncontrolled")
    os.makedirs(uncontrolled_save_dir, exist_ok=True)
    plot_loss_vs_iteration(trajectories_uncontrolled, args, save_dir=uncontrolled_save_dir)
    plot_final_loss_vs_iteration(trajectories_uncontrolled, args, save_dir=uncontrolled_save_dir)



if __name__ == "__main__":

    np.random.seed(42)

    dir_path = './applications/electricity_prediction/data/'
    xtrain = np.load(os.path.join(dir_path, 'xtrain.npy'))
    xval = np.load(os.path.join(dir_path, 'xval.npy'))
    xtest = np.load(os.path.join(dir_path, 'xtest.npy'))
    ytrain = np.load(os.path.join(dir_path, 'ytrain.npy'))
    yval = np.load(os.path.join(dir_path, 'yval.npy'))
    ytest = np.load(os.path.join(dir_path, 'ytest.npy'))

    llm_model, tokenizer = load_model('meta-llama/Llama-3.1-8B-Instruct', None)  # model that does shifting
    classifier_model = LogisticRegression(fit_intercept=False)

    # Run experiment
    args = Args()
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = ElectricityShiftSimulator(
        model=llm_model,
        tokenizer=tokenizer,
        classifier_model=classifier_model,
        data=[xtrain, ytrain, xval, yval],
    )
    loss_simulator = ElectricityLossSimulator(classifier_model)
    run_electricity_experiment(
        Z=[xtest, ytest],
        width_calculator=width_calculator,
        risk_measure=risk_measure,
        performativity_simulator=performativity_simulator,
        loss_simulator=loss_simulator,
        args=args,
        save_dir=f"./applications/electricity_prediction/figures/expected_loss/",
        num_iters=50
    )
