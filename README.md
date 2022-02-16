# Structural differences in adolescent brains can predict alcohol misuse ([medarxiv link](https://www.medrxiv.org/content/10.1101/2022.01.31.22269833v1))
**ABSTRACT:** Alcohol misuse during adolescence (AAM) has been linked with disruptive structural development of the brain and alcohol use disorder. Using machine learning (ML), we analyze the link between AAM phenotypes and adolescent brain structure (T1-weighted imaging and DTI) at ages 14, 19, and 22 in the IMAGEN dataset (n ∼1182). ML predicted AAM at age 22 from brain structure with a balanced accuracy of 78% on independent test data. Therefore, structural differences in adolescent brains could significantly predict AAM. Using brain structure at age 14 and 19, ML predicted AAM at age 22 with a balanced accuracy of 73% and 75%, respectively. These results showed that structural differences preceded alcohol misuse behavior in the dataset. The most informative features were located in the white matter tracts of the corpus callosum and internal capsule, brain stem, and ventricular CSF. In the cortex, they were spread across the occipital, frontal, and temporal lobes and in the cingulate cortex. Our study also demonstrates how the choice of the phenotype for AAM, the ML method, and the confound correction technique are all crucial decisions in an exploratory ML study analyzing psychiatric disorders with weak effect sizes such as AAM.

## Dataset
[IMAGEN dataset](https://doi.org/10.25720/p1ma-genq)

## Experiment 
[Overview](figures/overview.pdf)

![Full pipeline](figures/pipeline.pdf)

## Results
 
[Stage 1: ML exploration](figures/results_explore.pdf)

![Stage 2: Generalization](figures/results_infer.pdf)

## Code guide

1. [dataset-preprocessing.ipynb](dataset-preprocessing.ipynb) : IMAGEN data preprocessing performed before running the MLpipelines.
2. [dataset-statistics.ipynb](dataset-statistics.ipynb) : All data analysis performed prior to the experiments.
3. [MLpipelines/runMLpipelines.py](MLpipelines/runMLpipelines.py): Main pipeline run file. Change configurations within the file and run the script to generate a results file `run.csv`
4. [MLpipelines/plot_results.ipynb](MLpipelines/plot_results.ipynb): visualize the results obtained in `run.csv`
4. [generate_figures.ipynb](generate_figures.ipynb): code used to generate the final figures for the paper.

# Paper Reference: 
medarxiv: [https://www.medrxiv.org/content/10.1101/2022.01.31.22269833v1](https://www.medrxiv.org/content/10.1101/2022.01.31.22269833v1)
Roshan P. Rane, Evert F. de Man , JiHoon Kim , Dr. Kai Görgen , Mira Tschorn , Michael A. Rapp , Prof. Tobias Banaschewski , Prof. Arun Bokde , Sylvane Desrivieres , Prof. Herta Flor , Antoine Grigis , Hugh Garavan , Prof. Penny A. Gowland , Jean-Luc Martinot , Rüdiger Brühl , Marie-Laure P. Martinot , Eric Artiges , Dr. Frauke Nees , Dr. Dimitri Papadopoulos Orfanos , Herve Lemaitre , Tomas Paus , Luise Poustka , Dr. Juliane Fröhner , Lauren Robinson , Jeanne Winterer , Michael N. Smolka , Rob Whelan , Gunter Schumann , Prof. Henrik Walter , Prof. Andreas Heinz , Kerstin Ritter

Please cite our article if you find any part of this code useful for your ML experiments
```
@article{rane2022structural,
  title={Structural differences in adolescent brains can predict alcohol misuse},
  author={Rane, Roshan Prakash and de Man, Evert Ferdinand and Kim, JiHoon and G{\"o}rgen, Kai and Tschorn, Mira and Rapp, Michael A and Banaschewski, Tobias and Bokde, Arun LW and Desrivi{\`e}res, Sylvane and Flor, Herta and others},
  journal={medRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
