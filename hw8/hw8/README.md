::: centering
Assignment: Exploration and Offline Reinforcement Learning\
:::

# Introduction

This assignment requires you to implement and evaluate a pipeline for
exploration and offline learning. You will first implement an
DQN and collect
data using this reinforcement learning procedure, then perform offline training on
the data collected via DQN using conservative Q-learning (CQL) and
finally finetune the resulting policy in the environment. This
assignment would be easier to run on a CPU as we will be using gridworld
domains of varying difficulties to train our agents.

> Part 1 of this assignment requires you to implement and evaluate
> Q-learning for playing Atari games. The Q-learning algorithm was
> covered in lecture, and you will be provided with starter code. This
> assignment will run faster on a GPU, though it is possible to complete
> on a CPU as well. Note that we use convolutional neural network
> architectures in this assignment. Please start early! For references
> to this type of approach, see this
> [paper](https://arxiv.org/abs/1312.5602) and this
> [paper](https://arxiv.org/abs/1509.02971).

## Part 1: DQN
===========

We will be building on the code that we have implemented in the first
two assignments. All files needed to run your code are in the `hw3`
folder. Files to edit:

-   `infrastructure/rl_trainer.py`

-   `infrastructure/utils.py`

-   `policies/MLP_policy.py`

In order to implement deep Q-learning, you will be writing new code in
the following files:

-   `agents/dqn_agent.py`

-   `critics/dqn_critic.py`

-   `policies/argmax_policy.py`

There are two new package requirements (`opencv-python` and
`gym[atari]`) beyond what was used in the first two assignments; make
sure to install these with `pip install -r requirements.txt` if you are
running the assignment locally.

### Implementation 
--------------

The first phase of the assignment is to implement a working version of
Q-learning. The default code will run the `Ms. Pac-Man` game with
reasonable hyperparameter settings. Look for the `# TODO` markers in the
files listed above for detailed implementation instructions. You may
want to look inside `infrastructure/dqn_utils.py` to understand how the
(memory-optimized) replay buffer works, but you will not need to modify
it.

Once you implement Q-learning, answering some of the questions may
require changing hyperparameters, neural network architectures, and the
game, which should be done by changing the command line arguments passed
to `run_hw3_dqn.py` or by modifying the parameters of the `Args` class
from within the Colab notebook.

To determine if your implementation of Q-learning is correct, you should
run it with the default hyperparameters on the `Ms. Pac-Man` game for 1
million steps using the command below. Our reference solution gets a
return of 1500 in this timeframe. On Colab, this will take roughly 3 GPU
hours. If it takes much longer than that, there may be a bug in your
implementation.

To accelerate debugging, you may also test on `LunarLander-v3`, which
trains your agent to play Lunar Lander, a 1979 arcade game (also made by
Atari) that has been implemented in OpenAI Gym. Our reference solution
with the default hyperparameters achieves around 150 reward after 350k
timesteps, but there is considerable variation between runs, and without
the double-Q trick, the average return often decreases after reaching
150. We recommend using `LunarLander-v3` to check the correctness of
your code before running longer experiments with `MsPacman-v0`.

### Evaluation
----------

Once you have a working implementation of Q-learning, you should prepare
a report. The report should consist of one figure for each question
below. You should turn in the report as one PDF and a zip file with your
code. If your code requires special instructions or dependencies to run,
please include these in a file called `README` inside the zip file.
Also, provide the log file of your run on gradescope named as
`pacman_1.csv`.

#### Question 1: basic Q-learning performance (DQN).

Include a learning curve plot showing the performance of your
implementation on `Ms. Pac-Man`. The x-axis should correspond to a
number of time steps (consider using scientific notation), and the
y-axis should show the average per-epoch reward as well as the best mean
reward so far. These quantities are already computed and printed in the
starter code. They are also logged to the `data`. Be sure to label the
y-axis, since we need to verify that your implementation achieves
similar reward as ours. You should not need to modify the default
hyperparameters in order to obtain good performance, but if you modify
any of the parameters, list them in the caption of the figure. The final
results should use the following experiment name:

``` {.bash language="bash"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=MsPacman-v0 env.exp_name=q1
```

#### Question 2: double Q-learning (DDQN).

Use the double estimator to improve the accuracy of your learned Q
values. This amounts to using the online Q network (instead of the
target Q network) to select the best action when computing target
values. Compare the performance of DDQN to vanilla DQN. Since there is
considerable variance between runs, you must run at least three random
seeds for both DQN and DDQN. You may use `LunarLander-v3` for this
question. The final results should use the following experiment names:

``` {.bash language="bash" breaklines="true"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_1 logging.seed=1
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_2 logging.seed=2
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_dqn_3 logging.seed=3
```

``` {.bash language="bash" breaklines="true"}
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_1 alg.double_q=true logging.seed=1
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_2 alg.double_q=true logging.seed=2
python run_hw3_ql.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q2_doubledqn_3 alg.double_q=true logging.seed=3
```

Submit the run logs for all the experiments above. In your report, make
a single graph that averages the performance across three runs for both
DQN and double DQN. See `scripts/read_results.py` for an example of how
to read the evaluation returns from Tensorboard logs.

#### Question 3: experimenting with hyperparameters.

Now, let's analyze the sensitivity of Q-learning to hyperparameters.
Choose one hyperparameter of your choice and run at least three other
settings of this hyperparameter in addition to the one used in Question
1, and plot all four values on the same graph. Your choice is what you
experiment with, but you should explain why you chose this
hyperparameter in the caption. Examples include (1) learning rates; (2)
neural network architecture for the Q network, e.g., number of layers,
hidden layer size, etc; (3) exploration schedule or exploration rule
(e.g. you may implement an alternative to $\epsilon$-greedy and set
different values of hyperparameters), etc. Discuss the effect of this
hyperparameter on performance in the caption. You should find a
hyperparameter that makes a nontrivial difference in performance. Note:
you might consider performing a hyperparameter sweep to get good results
in Question 1, in which case it's fine to just include the results of
this sweep for Question 3 as well while plotting only the best
hyperparameter setting in Question 1. The final results should use the
following experiment name:

``` {.bash language="bash" breaklines="true"}
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam1
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam2
python run_hw3_dqn.py alg.rl_alg=dqn env.env_name=LunarLander-v3 env.exp_name=q3_hparam3
```

You can replace `LunarLander-v3` with `PongNoFrameskip-v4` or
`MsPacman-v0` if you would like to test on a different environment.

# Part 2: Conservative Q-learning

The starter code for this assignment can be found at

::: centering
<https://github.com/milarobotlearningcourse/robot_learning/hw5>\
:::

We will be building on the code that we have implemented in the earlier assignments. All files
needed to run your code are in the `hw5` folder, but there will be some
blanks you will fill with your previous solutions. Places that require
previous code are marked with `# TODO` and are found in the following
files:

-   `infrastructure/utils.py`

-   `infrastructure/rl_trainer.py`

-   `policies/MLP_policy.py`

-   `policies/argmax_policy.py`

-   `critics/dqn_critic.py`

In order to implement CQL you will be writing new code in the
following files:

-   `critics/cql_critic.py`

-   `agents/explore_or_exploit_agent.py`

-   `policies/MLP_policy.py`

<figure>
<p><img src="traj (1).png" style="width:32.0%" alt="image" />  <img
src="traj copy.png" style="width:32.0%" alt="image" />  <img
src="traj.png" style="width:32.0%" alt="image" /></p>
<figcaption>Figures depicting the <code>easy</code> (left),
<code>medium</code> (middle) and <code>hard</code> (right)
environments.</figcaption>
</figure>

## Environments

Unlike previous assignments, we will consider some stochastic dynamics,
discrete-action gridworld environments and envs from OGBench in this assignment. The three
gridworld environments and OGBench env you will need for the graded part of this
assignment are of varying difficulty: `easy`, `medium` and `hard` [OGBench](https://arxiv.org/abs/2410.20092) *visual-cube-triple-play-v0* dataset. A
picture of these environments is shown below. The easy environment
requires following two hallways with a right turn in the middle. The
medium environment is a maze requiring multiple turns. The hard
environment is a four-rooms task which requires navigating between
multiple rooms through narrow passages to reach the goal location. We
also provide a very hard environment for the bonus (optional) part of
this assignment.

## Conservative Q-Learning (CQL) Algorithm

For the first portion of the offline RL part of this assignment, we will
implement the conservative Q-learning (CQL) algorithm that augments the
Q-function training with a regularizer that minimizes the soft-maximum
of the Q-values $\log \left( \sum_{a} \exp(Q(s, a)) \right)$ and
maximizes the Q-value on the state-action pair seen in the dataset,
$Q(s, a)$. The overall CQL objective is given by the standard TD error
objective augmented with the CQL regularizer weighted by $\alpha$:
$\alpha \left[\frac{1}{N}\sum_{i=1}^N \left(\log\left(\sum_{a} \exp(Q(s_i, a))\right) - Q(s_i, a_i) \right) \right]$.
You will tweak this value of $\alpha$ in later questions in this
assignment.

## Implementation

The first part in this assignment is to implement a working version of
Deep Q-Learning. The default code will run the `easy`
environment with reasonable hyperparameter settings. Look for the
`# TODO` markers in the files listed above for detailed implementation
instructions.

Once you implement DQN, answering some of the questions will require
changing hyperparameters, which should be done by changing the command
line arguments passed to `run_hw8_offrl.py` or by modifying the
parameters of the `Args` class from within the Colab notebook.

For the second part of this assignment, you will implement the
conservative Q-learning algorithm as described above. Look for the
`# TODO` markers in the files listed above for detailed implementation
instructions. You may also want to add additional logging to understand
the magnitude of Q-values, etc, to help debugging. Finally, you will
also need to implement the logic for switching between exploration and
exploitation, and controlling for the number of offline-only training
steps in the `agents/explore_or_exploit_agent.py` as we will discuss in
problems 2 and 3.

## Evaluation

Once you have a working implementation of CQL, you should
prepare a report. The report should consist of one figure for each
question below (each part has multiple questions). You should turn in
the report as one PDF and a zip file with your code. If your code
requires special instructions or dependencies to run, please include
these in a file called `README` inside the zip file.

## Problems

#### Part 2: Offline learning on exploration data.

Now that we have implemented RND for collecting exploration data that is
(likely) useful for performing exploitation, we will perform offline RL
on this dataset and see how close the resulting policy is to the optimal
policy. To begin, you will implement the conservative Q-learning
algorithm in this question which primarily needs to be added in
`critic/cql_critic.py` and you need to use the CQL critic as the
extrinsic critic in `agents/explore_or_exploit_agent.py`. Once CQL is
implemented, you will evaluate it and compare it to a standard DQN
critic.

**For the first sub-part of this problem**, you will write down the
logic for disabling data collection in
`agents/explore_or_exploit_agent.py` after exploitation begins and only
evaluate the performance of the extrinsic critic after training on the
data collected by the RND critic. To begin, run offline training at the
default value of `num_exploration_steps` which is set to 10000. Compare
DQN to CQL on the `medium` environment.

``` {.bash language="bash"}
# cql_alpha = 0 => DQN, cql_alpha = 0.1 => CQL

python run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0

python run_hw5_expl.py --env_name PointmassMedium-v0  --exp_name q2_cql
--use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1
```

Examine the difference between the Q-values on state-action tuples in
the dataset learned by CQL vs DQN. Does CQL give rise to Q-values that
underestimate the Q-values learned via a standard DQN? If not, why? To
answer this question, first you might find it illuminating to try the
experiment shown below, marked as a hint and then reason about a common
cause behind both of these phenomena.

**Hint:** Examine the performance of CQL when utilizing a transformed
reward function for training the exploitation critic. Do not change any
code in the environment class, instead make this change in\
`agents/explore_or_exploit_agent.py`. The transformed reward function is
given by:
$$\tilde{r}(s, a) = (r(s, a) + \text{shift}) \times \text{scale}$$ The
choice of shift and scale is up to you, but we used $\text{shift}=1$,
and $\text{scale}=100.$ On any one domain of your choice test the
performance of CQL with this transformed reward. Is it better or worse?
What do you think is the reason behind this difference in performance,
if any?

**For the second sub-part of this problem**, perform an ablation study
on the performance of the offline algorithm as a function of the amount
of exploration data. In particular vary the amount of exploration data
for atleast two values of the variable `num_exploration_steps` in the
offline setting and report a table of performance of DQN and CQL as a
function of this amount. You need to do it on the `medium` or `hard`
environment. Feel free to utilize the scaled and shifted rewards if they
work better with CQL for you.

``` {.bash language="bash"}
python run_hw5_expl.py --env_name *Chosen Env* --use_rnd
--num_exploration_steps=[5000, 15000] --offline_exploitation --cql_alpha=0.1
--unsupervised_exploration --exp_name q2_cql_numsteps_[num_exploration_steps]

python run_hw5_expl.py --env_name *Chosen Env* --use_rnd 
--num_exploration_steps=[5000, 15000] --offline_exploitation --cql_alpha=0.0
--unsupervised_exploration --exp_name q2_dqn_numsteps_[num_exploration_steps]
```

**For the third sub-part of this problem**, perform a sweep over two
informative values of the hyperparameter $\alpha$ besides the one you
have already tried (denoted as `cql_alpha` in the code; some potential
values shown in the run command below) to find the best value of
$\alpha$ for CQL. Report the results for these values in your report and
compare it to CQL with the previous $\alpha$ and DQN on the `medium`
environment. Feel free to utilize the scaled and shifted rewards if they
work better for CQL.

``` {.bash language="bash"}
python run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd
--unsupervised_exploration --offline_exploitation --cql_alpha=[0.02, 0.5]
--exp_name q2_alpha[cql_alpha]
```

Interpret your results for each part. Why or why not do you expect one
algorithm to be better than the other? Do the results align with this
expectation? If not, why?


**Evalute your CQL on OGBench**
Just like for the hw1, we will compare the algorithm over [OGBench](https://arxiv.org/abs/2410.20092) *visual-cube-triple-play-v0* dataset. Use the same dataset from hw1 that was shared. Inlcude the learning plot in your submission.

# Submitting the code and experiment runs

In order to turn in your code and experiment logs, create a folder that
contains the following:

-   A folder named `data` with all the experiment runs from this
    assignment. **Do not change the names originally assigned to the
    folders, as specified by `exp_name` in the instructions. Video
    logging is not utilized in this assignment, as visualizations are
    provided through plots, which are outputted during training.**

-   The `roble` folder with all the `.py` files, with the same names and
    directory structure as the original homework repository (excluding
    the `data` folder). Also include any special instructions we need to
    run in order to produce each of your figures or tables (e.g. "run
    python myassignment.py -sec2q1" to generate the result for Section 2
    Question 1) in the form of a README file.

As an example, the unzipped version of your submission should result in
the following file structure. **Make sure that the submit.zip file is
below 15MB and that they include the prefix `q1_`, `q2_`, `q3_`, etc.**

::: forest
for tree= font=, grow'=0, child anchor=west, parent anchor=south,
anchor=west, calign=first, edge path= (!u.south west) +(7.5pt,0) \|-
node\[fill,inner sep=1.25pt\] (.child anchor); , before typesetting
nodes= if n=1 insert before=\[,phantom\] , fit=band, before computing
xy=l=15pt, \[submit.zip \[data \[q1\...
\[events.out.tfevents.1567529456.e3a096ac8ff4\] \] \[q2\...
\[events.out.tfevents.1567529456.e3a096ac8ff4\] \] \[\...\] \] \[roble
\[agents \[explore_or_exploit_agent.py\] \[awac_agent.py\] \[\...\] \]
\[policies \[\...\] \] \[\...\] \] \[README.md\] \[\...\] \]
:::

If you are a Mac user, **do not use the default "Compress" option to
create the zip**. It creates artifacts that the autograder does not
like. You may use `zip -vr submit.zip submit -x "*.DS_Store"` from your
terminal.

Turn in your assignment on Gradescope. Upload the zip file with your
code and log files to **HW5 Code**, and upload the PDF of your report to
**HW5**.
