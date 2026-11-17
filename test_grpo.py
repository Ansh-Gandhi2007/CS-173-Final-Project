import pytest
from train_grpo import balanced_camel_case_reward

def test_exact_match():
    prompts = ["Convert 'vpc_id' to camelCase."]
    completions = ["vpcId"]
    ground_truth = ["vpcId"]
    rewards = balanced_camel_case_reward(prompts, completions, ground_truth)
    assert rewards[0] == 1.0

def test_partial_heuristics():
    prompts = ["Convert 'vpc_id' to camelCase."]
    completions = ["vpcid"] 
    ground_truth = ["vpcId"]
    rewards = balanced_camel_case_reward(prompts, completions, ground_truth)
    assert rewards[0] == 0.7

def test_failure_case():
    prompts = ["Convert 'vpc_id' to camelCase."]
    completions = ["vpc_id"] 
    ground_truth = ["vpcId"]
    rewards = balanced_camel_case_reward(prompts, completions, ground_truth)
    assert rewards[0] == 0.2