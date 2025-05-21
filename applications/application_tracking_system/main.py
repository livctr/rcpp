import logging
import random
import re
from typing import Any, Dict, List, Tuple

from langchain_core.runnables import Runnable
from langchain_community.llms import VLLM
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


from langchain_core.prompts import ChatPromptTemplate


def generate_applicant_data(model, n: int, p: float = 0.5, p1: float = 0.8, p2: float = 0.2, shuffle: bool = False) -> Tuple[List[str], List[int]]:
    """
    Generates a dataset of data science applicant summaries and their labels.

    Args:
        n (int): The total number of applicant summaries to generate.
        p (float): The probability (0.0 to 1.0) of generating applicant type I.
        p1 (float): The probability that applicant type I is strong.
        p2 (float): The probability that applicant type II is strong.
        shuffle (bool): Whether to shuffle the generated data (X and y) together.
        temperature (float): The temperature setting for the model.

    Returns:
        tuple: A tuple containing two lists:
            - X (list): A list of applicant summary strings.
            - y (list): A list of labels (1 for strong, 0 for typical).
    """
    # Define the prompts
    strong_applicant_prompt = ChatPromptTemplate([
        (
            "system",
            "You are a data science professional articulating your suitability for a challenging data science role in 100 words. "
        ),
        (
            "human",
            "Articulate your key contributions to past data-driven projects, focusing on how your "
            "analytical approach led to measurable improvements or novel insights. "
            "Feel free to mention tools and methodologies you used, your ability to solve complex problems, "
            "or your capacity to communicate complex findings to non-technical stakeholders. "
            "Begin with 'As'. Do not exceed the 100-word limit, else you will not be hired."
            "ONLY OUTPUT YOUR RESPONSE, DO NOT OUTPUT ANY OTHER TEXT!"
        )
    ])
    typical_applicant_prompt = ChatPromptTemplate([
        (
            "system",
            "You are an aspiring data scientist articulating your qualifications for an entry-level or junior role in 100 words. "
        ),
        (
            "human",
            "Articulate your key qualifications, including relevant coursework, projects, or prior experience. "
            "Highlight your enthusiasm for learning and applying data analysis techniques to real-world problems. "
            "Mention your comfort with standard data tools and a desire to grow your skills within a collaborative team environment. "
            "Begin with 'As'. Do not exceed the 100-word limit, else you will not be hired."
            "ONLY OUTPUT YOUR RESPONSE, DO NOT OUTPUT ANY OTHER TEXT!"
        )
    ])

    X = []  # List to store applicant summaries
    y = []  # List to store labels (1 for strong, 0 for typical)

    # Create chains for each prompt
    strong_chain: Runnable = strong_applicant_prompt | model
    typical_chain: Runnable = typical_applicant_prompt | model

    # Determine the number of strong and typical applicants
    num_strong = round(n * p)
    num_typical = n - num_strong

    # Generate strong applicants
    if num_strong > 0:
        logging.info(f"Generating {num_strong} strong applicants...")
        strong_inputs = [{} for _ in range(num_strong)] # Empty dictionaries as input since prompts are self-contained
        strong_results = strong_chain.batch(strong_inputs)
        X.extend(strong_results)
        y.extend([1] * num_strong)

        logging.info("=" * 40)
        logging.info(f"generate_applicant_data (Label: 'Strong')")
        logging.info("=" * 40)
        logging.info(f"{strong_results[0]}")
        logging.info("=" * 40)

    # Generate typical applicants
    if num_typical > 0:
        logging.info(f"Generating {num_typical} typical applicants...")
        typical_inputs = [{} for _ in range(num_typical)]
        typical_results = typical_chain.batch(typical_inputs)
        X.extend(typical_results)
        y.extend([0] * num_typical)

        logging.info("=" * 40)
        logging.info(f"generate_applicant_data (Label: 'Typical')")
        logging.info("=" * 40)
        logging.info(f"{typical_results[0]}")
        logging.info("=" * 40)

    # Combine and shuffle if requested
    if shuffle:
        combined_data = list(zip(X, y))
        random.shuffle(combined_data)
        X, y = zip(*combined_data) # Unzip the shuffled data
        X = list(X)
        y = list(y)

    X = np.array(X)
    y = np.array(y)
    logging.info("Generation complete!")
    return X, y


def create_ats_scorer(scorer_model) -> Runnable:
    """
    Creates an LLM-based ATS scorer chain.
    """
    ats_prompt_template = ChatPromptTemplate([
        (
            "system",
            "You are a seasoned Hiring Manager for a Data Scientist role. Rate the applicant's summary (0-100).\n"
            "Indicators of a strong candidate: advanced ML (deep learning, NLP, CV), MLOps/deployment, leadership, quantifiable impact, specialized tools (TF, PyTorch, Spark, cloud), PhD/research.\n"
            "Indicators of a typical candidate: foundational skills (data cleaning, basic EDA), common libraries (Pandas, Scikit-learn), general project experience, academic focus.\n"
            "Score: 100 (top-tier), 80-99 (very strong), 60-79 (good/typical), 40-59 (decent), <40 (needs experience).\n"
            "\n"
            "Applicant Summary:\n"
            "{applicant_summary}\n"
            "Please output the following. If you don't, you will get fired! Be critical.\n"
            "Score: [0-100]\n"
            "Justification: [One-sentence justification. Be super concise!]"
        )
    ])
    return ats_prompt_template | scorer_model


def run_ats_on_batch(
    scorer_model, # Pass the LLM model instance directly
    applicant_summaries: List[str],
    labels: List[int], # For comparison, not used by ATS for scoring
) -> List[Tuple[int, str]]:
    """
    Runs the ATS system on a batch of applicant summaries using true LLM batching.

    Args:
        model: The LLM model instance to use for scoring.
        applicant_summaries (List[str]): A list of applicant summary strings.
        labels (List[int]): The true labels (1 for strong, 0 for typical) for comparison.

    Returns:
        List[Tuple[int, str]]: A list of tuples, each containing (score, justification).
    """
    ats_scorer_chain = create_ats_scorer(scorer_model) # Create the chain with the model

    # Prepare inputs for batch processing
    batch_inputs = [{"applicant_summary": summary} for summary in applicant_summaries]

    logging.debug(f"Running ATS on {len(applicant_summaries)} applicants...")
    raw_ats_responses = ats_scorer_chain.batch(batch_inputs) # This is the true batch call

    results = []
    for i, response_text in enumerate(raw_ats_responses):
        score_match = re.search(r"Score:\s*(\d+)", response_text)
        justification_match = re.search(r"Justification:\s*(.*)", response_text, re.DOTALL)

        score = 0
        justification = "Error parsing score or justification."

        if score_match:
            try:
                score = int(score_match.group(1))
                score = max(0, min(100, score))
            except ValueError:
                pass

        if justification_match:
            justification = justification_match.group(1).strip()

        results.append((score, justification))
    
    strong_idxs = np.where(np.array(labels) == 1)[0]
    typical_idxs = np.where(np.array(labels) == 0)[0]
    if len(strong_idxs) > 0:
        strong_idx = strong_idxs[0]
        logging.debug("=" * 40)
        logging.debug(f"Strong applicant example: {applicant_summaries[strong_idx]} (Label: 'Strong')")
        logging.debug("=" * 40)
        logging.debug(f"Strong applicant ATS score: {results[strong_idx][0]}")
        logging.debug(f"Justification: {results[strong_idx][1]}")
        logging.debug("=" * 40)
    if len(typical_idxs) > 0:
        typical_idx = typical_idxs[0]
        logging.debug("=" * 40)
        logging.debug(f"Typical applicant example: {applicant_summaries[typical_idx]} (Label: 'Typical')")
        logging.debug("=" * 40)
        logging.debug(f"ATS Score: {results[typical_idx][0]}")
        logging.debug(f"Justification: {results[typical_idx][1]}")
        logging.debug("=" * 40)

    return np.array(results)


def create_summary_modifier(model) -> Runnable:
    """
    Creates an LLM-based chain for modifying applicant summaries.
    """
    modification_prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an AI career coach helping a data scientist improve their resume summary.\n"
            "You have analyzed their current summary and received feedback from an ATS system.\n"
            "\n"
            "Original Summary:\n"
            "{original_summary}\n"
            "\n"
            "ATS Score: {ats_score}\n"
            "ATS Justification: {ats_justification}\n"
            "\n"
            "Modification Intensity (0 = no change, 1 = maximum change): {modification_intensity:.2f}\n"
            "\n"
            "Based on the ATS feedback and the modification intensity:\n"
            "- If intensity is 0, return the original summary verbatim.\n"
            "- If intensity is high (e.g., near 1), make significant changes to address the ATS justification,\n"
            "  emphasizing business impact, advanced techniques, leadership, and quantifiable results.\n"
            "  Ensure the new summary sounds like a very strong candidate.\n"
            "- If intensity is moderate, make thoughtful, subtle improvements focusing on clarity and incorporating\n"
            "  stronger phrasing suggested by the ATS justification.\n"
            "\n"
            "Begin with 'As'. Only output the 100-word revised summary. Else you will get fired!\n"
        )
    ])
    return modification_prompt_template | model


def modify_applicant_summaries_batch(
    original_summaries: List[str],
    labels: List[int], # For comparison, not used by ATS for scoring
    ats_feedback: List[Tuple[int, str]], # List of (score, justification)
    lambda_param: int, # lambda between 0 and 100
    summary_modifier_chain: Runnable,
) -> List[str]:
    """
    Modifies a list of applicant summaries based on ATS feedback and a lambda parameter.

    Args:
        original_summaries (List[str]): A list of the applicant's original summaries.
        labels (List[int]): A list of labels (1 for strong, 0 for typical) for comparison.
        ats_feedback (List[Tuple[int, str]]): A list of tuples, where each tuple
                                               contains (ats_score, ats_justification)
                                               for the corresponding summary.
        lambda_param (int): Controls modification extent (0=max change, 100=no change).
        summary_modifier_chain (Runnable): The LangChain Runnable for summary modification.

    Returns:
        List[str]: A list of the modified applicant summaries.
    """

    # Convert lambda to modification intensity (0 = no change, 1 = max change)
    modification_intensity = max(0., 1. - lambda_param)

    if modification_intensity <= 1e-4:
        # If no modification is desired, return original summaries directly
        return original_summaries

    # Prepare inputs for batch processing
    batch_inputs: List[Dict[str, Any]] = []
    for i, summary in enumerate(original_summaries):
        ats_score, ats_justification = ats_feedback[i]
        batch_inputs.append({
            "original_summary": summary,
            "ats_score": ats_score,
            "ats_justification": ats_justification,
            "modification_intensity": modification_intensity
        })

    logging.debug(f"Modifying {len(original_summaries)} summaries with Lambda = {lambda_param}...")
    modified_responses = summary_modifier_chain.batch(batch_inputs)
    
    # Strip whitespace from each modified summary
    modified_summaries = [res.strip() for res in modified_responses]

    logging.debug("Modification complete!")
    strong_idxs = np.where(np.array(labels) == 1)[0]
    typical_idxs = np.where(np.array(labels) == 0)[0]
    if len(strong_idxs) > 0:
        strong_idx = strong_idxs[0]
        logging.debug("=" * 40)
        logging.debug(f"Strong applicant example: {original_summaries[strong_idx]} (Label: 'Strong')")
        logging.debug("=" * 40)
        logging.debug(f"Strong applicant modified summary: {modified_summaries[strong_idx]}")
        logging.debug("=" * 40)
    if len(typical_idxs) > 0:
        typical_idx = typical_idxs[0]
        logging.debug("=" * 40)
        logging.debug(f"Typical applicant example: {original_summaries[typical_idx]} (Label: 'Typical')")
        logging.debug("=" * 40)
        logging.debug(f"Strong applicant modified summary: {modified_summaries[typical_idx]}")
        logging.debug("=" * 40)

    return np.array(modified_summaries)






class Args:
    def __init__(self):
        self.alpha = 0.2         # risk control level
        self.tightness = 0.072    # tightness parameter, may throw error if too low
        self.delta = 0.1        # failure probability or confidence parameter
        self.tau = 1.0           # safety parameter
        self.N = 2000            # number of samples in cohort
        self.lambda_min = 0.0
        self.lambda_safe = 0.2   # maximum value for lambda
        self.ell_max = 1.0
        self.gamma = 0.0


from rcpp.loss_simulator import LossSimulator
from rcpp.width_calculator import CLTWidth
from rcpp.risk_measure import MeanRiskMeasure
from rcpp.performativity_simulator import PerformativitySimulator


class ATSPerformativitySimulator(PerformativitySimulator):

    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature

    def simulate_shift(self,
                       Z_base: List,
                       Z_prev: List,
                       lambda_: float,
                       gamma: float) -> List:
        prev_temp = self.model.temperature
        self.model.temperature = self.temperature
        applicant_summaries, first_ats_feedback, _, labels = Z_base
        shifted_summaries = modify_applicant_summaries_batch(
            applicant_summaries,
            labels,
            first_ats_feedback,
            lambda_,
            create_summary_modifier(self.model)
        )

        self.model.temperature = 0.0
        cur_ats_feedback = run_ats_on_batch(self.model, shifted_summaries, labels)
        self.model.temperature = prev_temp
        return shifted_summaries, first_ats_feedback, cur_ats_feedback, labels



class ATSLossSimulator(LossSimulator):

    def __init__(self, scorer_model, temperature=0.0):
        self.scorer_model = scorer_model
        self.temperature = temperature

    def calc_loss(self, Z: List, lambda_: float, do_new_sample: bool = True):
        prev_temp = self.scorer_model.temperature
        self.scorer_model.temperature = self.temperature
        _, _, cur_ats_feedback, labels = Z
        ats_scores = []
        for ats_score, _ in cur_ats_feedback:
            try:
                ats_scores.append(int(ats_score))
            except ValueError:
                ats_scores.append(0)
        avg_ats_scores = np.mean(ats_scores)
        logging.info(f"Average ATS Score: {avg_ats_scores} | Lambda: {lambda_}")

        norm_scores = np.array(ats_scores) / 100.0  # Normalize scores to [epsilon, 1-epsilon]
        cont_indicator = (1 - lambda_ >= norm_scores)
        self.scorer_model.temperature = prev_temp
        return cont_indicator * (np.array(labels, dtype=np.int8) == 1)


from rcpp.main import run_trajectory
import os
import json
from functools import wraps

CACHE_JSON_FILE = './applications/application_tracking_system/data/applicant_data_cache.json'

def persist_json_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # A simple way to handle arguments for file naming, if needed
        # For this example, we'll just save the result of one specific call
        # A more robust solution would involve hashing args for unique file names
        # or storing a dictionary of results keyed by argument hashes in one file.

        if os.path.exists(CACHE_JSON_FILE):
            logging.info(f"Loading data from persistent JSON cache: {CACHE_JSON_FILE}")
            with open(CACHE_JSON_FILE, 'r') as f:
                data = json.load(f)
                return np.array(data['summaries']), np.array(data['labels'])
        else:
            logging.info("No persistent JSON cache found. Generating data...")
            applicant_summaries, labels = func(*args, **kwargs)
            data_to_save = {
                'summaries': applicant_summaries.tolist(),
                'labels': labels.tolist()
            }
            os.makedirs(os.path.dirname(CACHE_JSON_FILE), exist_ok=True)
            with open(CACHE_JSON_FILE, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            logging.info(f"Data saved to persistent JSON cache: {CACHE_JSON_FILE}")
            return applicant_summaries, labels
    return wrapper

# Apply the JSON persistence decorator, reads the same file regardless of n and p
@persist_json_cache
def get_applicant_data_json_persisted(model, **kwargs):
    return generate_applicant_data(model, **kwargs) # Re-use dummy generation


if __name__ == "__main__":

    args = Args()
    save_dir = "./applications/application_tracking_system/figures/expected_loss/"

    vllm = VLLM(model="meta-llama/Llama-3.2-3B-Instruct", top_p=0.95)
    vllm.max_new_tokens = 250
    vllm.temperature = 0.4
    applicant_summaries, labels = get_applicant_data_json_persisted(vllm, n=4000, p=0.5, p1=0.8, p2=0.2, shuffle=True)
    ats_scorer_chain = create_ats_scorer(vllm)
    vllm.temperature = 0.0
    ats_feedback = run_ats_on_batch(vllm, applicant_summaries, labels)

    args = Args()
    width_calculator = CLTWidth(args.alpha, args.ell_max, tol=1e-5)
    risk_measure = MeanRiskMeasure()
    performativity_simulator = ATSPerformativitySimulator(vllm, temperature = 0.4)
    loss_simulator = ATSLossSimulator(vllm, temperature = 0.0)

    trajectories = []
    for _ in range(20):
        idx = np.random.choice(len(applicant_summaries), size=args.N, replace=False)
        trajectory = run_trajectory(
            [applicant_summaries, ats_feedback, ats_feedback, labels],
            idx,
            width_calculator,
            risk_measure,
            performativity_simulator,
            loss_simulator,
            args,
        )
        trajectories.append(trajectory)

        logging.info(f"Trajectory: {trajectory}")

        # Save trajectory to pickle
        import pickle
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"trajectories.pkl"), 'wb') as f:
            pickle.dump(trajectories, f)
